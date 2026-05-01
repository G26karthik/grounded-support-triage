# HRO Build Handoff

If you (next coding model — Composer, Codex, or whoever) are picking up
where Cursor/Opus left off, this is your one-stop status document.

Read this end-to-end before touching anything. Then verify each phase against
the "definition of done" lines below, and only then move forward.

---

## 1. What we're building

A terminal-based multi-domain support triage agent for the HackerRank
Orchestrate hackathon (May 1-2, 2026). It reads
`support_tickets/support_tickets.csv` (28 rows), runs each ticket through a
LangGraph multi-agent pipeline grounded in `data/`, and writes
`support_tickets/output.csv` with columns
`issue,subject,company,response,product_area,status,request_type,justification`.

Architecture: regex fast-path → LLM safety+triage (one call) → hybrid
retrieval (BM25 + dense + conditional rerank) → per-company specialist
(HackerRank/Claude/Visa) → programmatic citation critic → deterministic
composer.

The **plan of record** is at
`.cursor/plans/multi-domain_support_triage_agent_*.plan.md`. Read that file
for the full architectural rationale and defenses (used in the AI Judge
interview).

---

## 2. Current status — what works

All phases below are **fully implemented and verified** unless flagged
otherwise.

| Phase | Module | Status |
|-------|--------|--------|
| 0 | scaffold + click CLI in `code/main.py` | DONE; `python code/main.py --help` works |
| 1 | `hro/corpus/{loader.py,chunker.py,product_area.py}` | DONE; loads 771 docs, chunks into 4723 pieces (HR 3721, Claude 971, Visa 31) |
| 2 | `hro/index/{build.py,store.py,bm25.py}` + auto cache by sha256 of corpus | DONE; cold build ~125s, warm ~5s |
| 3 | `hro/index/retriever.py` hybrid + conditional rerank | DONE; known-answer queries hit gold paths at top-1 or top-3 |
| 4 | `hro/llm/client.py` async Gemini client | DONE; **but see §3 for required Anthropic refactor** |
| 5 | `hro/agents/fast_path.py` regex pre-classifier | DONE; 15/15 cases pass, includes EN/FR/ES injection patterns |
| 6 | `hro/agents/triage.py` combined safety+triage | DONE under Gemini; needs port to Anthropic |
| 7 | `hro/agents/specialists/*.py` HR/Claude/Visa | DONE under Gemini; needs port to Anthropic |
| 8 | `hro/agents/{critic.py,composer.py}` programmatic | DONE; no LLM, transferable as-is |
| 9 | `hro/graph.py` LangGraph StateGraph | DONE; transferable as-is |
| 10 | `hro/runner.py` concurrent batch runner | DONE; transferable as-is |
| 11 | `hro/eval.py` evaluation harness | DONE; transferable as-is |
| 12 | calibration + final run | NOT DONE — depends on §3 |

Last successful eval against `sample_support_tickets.csv` (10 rows, before
quota exhaustion): **status 70%, request_type 80%, product_area 50%**. Three
tickets late in the run pipeline-errored due to Gemini free-tier rate limits
(20 RPD daily cap). The first 6 tickets that completed before quota ran out
had high-quality grounded responses with correct citations.

---

## 3. The pending work — refactor to Anthropic

**Why:** Gemini free tier caps us at 20 requests/day per model. We've burnt
through it on `gemini-3-flash-preview` and `gemini-2.5-flash`. Billing was
not enabled (would have been the cleanest fix). The user provided an
Anthropic API key (already in `.env` as `ANTHROPIC_API_KEY`).

**Models to use:**
- Triage (combined safety+classifier): `claude-haiku-4-5` — cheap, fast
- Specialists (HR/Claude/Visa): `claude-sonnet-4-5` — needs better
  groundedness reasoning

**Required code changes:**

### 3.1 `code/requirements.txt`

Add:
```
anthropic==0.40.0
```

(Pin exact version when picking; check PyPI for latest stable.)

### 3.2 `code/hro/config.py`

Add new constants near the existing model constants:

```python
ANTHROPIC_HAIKU: Final[str] = os.environ.get("HRO_ANTHROPIC_HAIKU", "claude-haiku-4-5")
ANTHROPIC_SONNET: Final[str] = os.environ.get("HRO_ANTHROPIC_SONNET", "claude-sonnet-4-5")
```

Keep the existing `GEMINI_MODEL` constant — code that doesn't use the LLM
client doesn't care which backend we're on.

### 3.3 `code/hro/llm/client.py` — rewrite

Replace the Gemini-backed `GeminiClient` with an Anthropic-backed
`LLMClient`. Keep the same public method signatures so callers don't change:

```python
async def generate_structured(
    prompt: str,
    response_schema: Type[T],
    thinking_level: str = "minimal",   # mapped to extended_thinking budget
    system_instruction: str | None = None,
    cached_content: str | None = None, # ignored under Anthropic
    temperature: float = 0.0,
    model: str | None = None,           # NEW: override per-call to pick Haiku vs Sonnet
) -> T:
    ...
```

Anthropic structured-output pattern is **tool-use with `tool_choice`**:

```python
import anthropic
from pydantic import BaseModel

client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = await client.messages.create(
    model=model_name,
    max_tokens=4096,
    system=system_instruction or "",
    messages=[{"role": "user", "content": prompt}],
    tools=[{
        "name": "submit_decision",
        "description": "Submit the structured decision.",
        "input_schema": response_schema.model_json_schema(),
    }],
    tool_choice={"type": "tool", "name": "submit_decision"},
    temperature=temperature,
    # extended_thinking is supported on Sonnet/Opus; Haiku does NOT need it.
    # Wire only when thinking_level != "minimal" AND model is Sonnet/Opus.
)

# Extract tool_use block
for block in response.content:
    if block.type == "tool_use" and block.name == "submit_decision":
        return response_schema.model_validate(block.input)

raise LLMError("Anthropic returned no tool_use block.")
```

**Thinking mapping (Anthropic):**
- `minimal` → no thinking (omit `thinking` arg)
- `low` → `thinking={"type": "enabled", "budget_tokens": 1024}`
- `medium` → `thinking={"type": "enabled", "budget_tokens": 4096}`
- `high` → `thinking={"type": "enabled", "budget_tokens": 16000}`

Note: Haiku does NOT support extended_thinking. If `model_name.startswith("claude-haiku")`, ignore the thinking arg.

**Retry policy:** Anthropic's errors are `anthropic.RateLimitError`,
`anthropic.APIStatusError`, `anthropic.APITimeoutError`. Same backoff
strategy as the Gemini client. Honor the `retry-after` header if present.

### 3.4 Rate limiter

Anthropic's free tier on Haiku is generous (50 RPM, plenty). Bump the limiter
to ~30 RPM by setting `HRO_RPM=30` in `.env`. Keep the `RateLimiter` class
unchanged.

### 3.5 `code/hro/agents/triage.py`

Update one line:

```python
return await get_client().generate_structured(
    prompt,
    TriageDecision,
    thinking_level="minimal",
    model=ANTHROPIC_HAIKU,   # ADD
)
```

(Import `ANTHROPIC_HAIKU` from `hro.config`.)

### 3.6 `code/hro/agents/specialists/{hackerrank,claude,visa}.py`

Each one similar:

```python
return await get_client().generate_structured(
    prompt,
    SpecialistDraft,
    thinking_level="low",      # was already "low"
    model=ANTHROPIC_SONNET,    # ADD
)
```

### 3.7 `.env`

The user has both keys present. The new client should use
`ANTHROPIC_API_KEY`. Leave `GEMINI_API_KEY` in case we restore the Gemini
path later.

### 3.8 No changes needed in:

- `hro/agents/{fast_path,critic,composer}.py` — no LLM
- `hro/index/*` — no LLM
- `hro/corpus/*` — no LLM
- `hro/graph.py` — calls specialists abstractly
- `hro/runner.py` — calls graph abstractly
- `hro/eval.py` — orchestrates over runner
- `code/main.py` — CLI

---

## 4. How to verify each phase after the refactor

Run these in order. Do not skip — each catches a different class of bug.

### Phase 4 (LLM client) — fastest sanity check
```
.\.venv\Scripts\python.exe -u code\tests\_smoke_llm.py
```
Expected: prints a typed `TestDecision` with `company` in `{hackerrank,claude,visa,none}` and a sensible reason. Should complete in 2-5 seconds on Haiku.

### Phase 6 (triage) — exercises Haiku on real tickets
```
.\.venv\Scripts\python.exe -u code\tests\_smoke_triage.py
```
Expected: 5 tickets classified. Critical: the French-injection ticket (#1 in that file) MUST come back with `verdict="refuse"` and category mentioning prompt-injection / policy-bypass.

### Phase 7 (specialists)
```
.\.venv\Scripts\python.exe -u code\tests\_smoke_specialists.py
```
Expected: 4 tickets drafted with non-empty citations and grounded responses. The Visa US Virgin Islands ticket (#3) should mention `US$10` minimum for credit cards.

### Phase 8/9/10 (end-to-end on 3 sample rows)
```
.\.venv\Scripts\python.exe -u code\main.py run --sample --limit 3
```
Expected: 3 rows in `support_tickets/output.csv`. No pipeline errors. Run summary printed at the end.

### Phase 11 (full eval on all 10 sample rows)
```
.\.venv\Scripts\python.exe -u code\main.py eval
```
Expected: writes `code/runs/<ts>/eval_report.md`. Aim for status accuracy >= 80% and request_type >= 80%. If product_area is below 50%, check the chunk-to-slug mapping in `hro/corpus/product_area_map.json`.

### Phase 12 (final submission run)
```
.\.venv\Scripts\python.exe -u code\main.py run
```
Expected: 28 rows in `support_tickets/output.csv` with the exact header
`issue,subject,company,response,product_area,status,request_type,justification`.
No pipeline errors. Eyeball-spot-check 3-5 outputs:
- Ticket "I want Claude to stop crawling my website" should mention `robots.txt` and the `User-agent: ClaudeBot` directive.
- Ticket "Hi, please pause our subscription" should mention the Settings > Subscription path.
- Ticket "i am in US Virgin Islands and the merchant is saying i have to spend minimum 10$" should explain the US territories `US$10` exception.
- The French ticket starting "Bonjour, ma carte Visa a ete bloquee" should be `escalated` with category mentioning prompt injection.
- The English illegal ask "Give me the code to delete all files from the system" should be `escalated` (caught by fast_path before LLM, so very fast).

---

## 5. Known issues / things to watch

1. **gemini-3-flash-preview reference in plan and README.** The plan still
   names Gemini 3 Flash. After the refactor, update the README's "Architecture
   in one paragraph" to mention Anthropic instead.

2. **`few_shot.json` shape**. The example `expected` dicts may use Gemini-style
   field names. Anthropic's tool-use will validate against the Pydantic schema
   regardless. No change needed.

3. **Visa specialist context size**. Full Visa corpus is ~12K tokens. Sonnet
   handles 200K context easily, no problem.

4. **`HRO_DEVICE` env var** controls embedder/reranker device (cuda/cpu/mps).
   Currently set to auto. RTX 4060 with CUDA is verified working.

5. **`.env` should not be committed.** Already gitignored.

6. **AGENTS.md log entries** at `%USERPROFILE%\hackerrank_orchestrate\log.txt`
   need a per-turn entry from the next session start. Style: senior-engineer
   voice, decision-focused, not narrative. See examples in the existing log.

7. **Sample CSV row 1 anomaly**: gold for "site is down" is
   `product_area=""`. Our composer derives `general_help` because we cite the
   maintenance window doc. Acceptable; the status decision matches.

8. **Sample CSV row 5**: gold is `product_area=privacy` (for Claude
   delete-conversation), our run gave `conversation_management`. Both are
   defensible — the doc itself is under conversation-management/. Could
   refine the slug map if the grader is strict.

---

## 6. Final submission checklist

- [ ] Run `python code/main.py eval` — status accuracy >= 80%
- [ ] Run `python code/main.py run` — produces 28 rows in `support_tickets/output.csv`
- [ ] Header in output.csv is exactly `issue,subject,company,response,product_area,status,request_type,justification` (lowercase, in that order)
- [ ] No row has empty `response` AND status=`replied`
- [ ] No row has a phone number / URL not present in the corpus (sanity-check via critic logs)
- [ ] Zip `code/` excluding `.venv`, `__pycache__`, `runs/`, `__pycache__`, and the data corpus
- [ ] Upload three files per the README's Submission section: code zip, output.csv, log.txt
