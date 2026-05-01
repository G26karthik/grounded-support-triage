# Project deep-dive (architecture & history)

Single narrative from kickoff through shipping: **what** we built, **why** each choice exists, and **what to admit** in interviews. The **public landing page** for recruiters and hiring managers is the repository root [`README.md`](./README.md); this file goes one level deeper.

---

## 1. Problem we solved

Build a **terminal-based**, **corpus-grounded** multi-company support triage agent:

- **Input:** `support_tickets/support_tickets.csv` (HackerRank, Claude, Visa, or unscoped rows).
- **Output:** `support_tickets/output.csv` with: `status`, `product_area`, `response`, `justification`, `request_type` (plus passthrough issue/subject/company).
- **Hard rules:** No live web browsing; answers must come from local `data/` markdown only; safety before LLMs; intelligent escalation when the corpus or policy cannot support a grounded reply; API keys **only** from environment variables.

**Knowledge base:** `data/hackerrank/`, `data/claude/`, `data/visa/` — **not** derived from `support_tickets.zip`. The zip bundles the same ticket CSVs as in the repo; the agent reads the extracted CSV on disk.

---

## 2. Architecture (what actually shipped)

**Runtime:** LangGraph `StateGraph` — explicit nodes and edges, not an opaque agent loop.

**Flow (conceptual):**

1. **Fast path (regex / heuristics, 0 LLM calls):** Trivial acknowledgements, empty tickets, hard injection patterns, clearly illegal requests → immediate canned or escalate path.
2. **Triage (1× LLM, structured output):** Single call combines **safety** + **routing**: allow/refuse/escalate, `request_type`, `scope`, `inferred_company`, English `retrieval_query`, language, intent summary. Model: **Claude Haiku** (volume + latency).
3. **Retrieve:** Hybrid **BM25 + dense embeddings** (BGE-small), z-score fusion, optional **cross-encoder rerank** (mxbai-rerank-xsmall) when scores are close. **Visa:** small corpus → full context to the specialist (and critic still checks citations). **HackerRank / Claude:** top-k chunks only.
4. **Specialist (1× LLM per non-terminal path):** Company-specific prompt + retrieved text. Model: **Claude Sonnet** (stronger grounding for drafted answers).
5. **Critic (no LLM):** Programmatic checks: cited `chunk_id`s must be in the retrieved set; “hard facts” in the reply (URLs, phones, emails, dollar amounts) must appear in cited text (with sensible normalization).
6. **Composer (no LLM):** Assembles final CSV row; if critic fails → escalate with safe customer text. **`product_area`** is derived from corpus path via a **prefix map** (`product_area_map.json`) plus a few **deterministic rules** (see §6).

**Observability:** Each run gets `code/runs/<utc>/trace.jsonl` + `summary.md`; optional eval report on the labelled sample.

**Concurrency:** `asyncio` + semaphore; sliding-window **rate limiter** to avoid provider 429s.

**LLM abstraction:** `LLMClient` with `HRO_BACKEND=anthropic` (default) or `gemini`. Structured output: Anthropic **forced tool use** with JSON schema from Pydantic; Gemini uses native schema when enabled.

---

## 3. Why LangGraph / “enterprise” shape

The hackathon rewards **clear orchestration**, **auditability**, and **defensible escalation**. A LangGraph-style graph makes every step **named, logged, and testable** — closer to what you’d ship behind a real queue than a single monolithic prompt. It also maps cleanly to the “AI judge” story: you can point to exact nodes when asked about safety, retrieval, or failures.

---

## 4. Pivot: Gemini → Anthropic

**Original spec** aimed at **Gemini 3 Flash** (thinking, 1M context, google-genai SDK).

**What happened:** Google AI Studio **free tier** hit a **severe daily request cap** (on the order of tens of requests per day), incompatible with iterative development and full batch runs (many LLM calls per ticket).

**Decision:** Default backend switched to **Anthropic** (Haiku triage + Sonnet specialists). **Gemini path kept** behind `HRO_BACKEND=gemini` for portability.

**Lesson for interviews:** We traded “model on the brief” for **reliability of execution** under real API quotas — a pragmatic production tradeoff, documented in README and code comments.

---

## 5. “Harness engineering” (latency + cost)

Goals: fewer calls, predictable failures, honest escalation.

- **0 LLM calls** for many trivial / malicious inputs (fast path).
- **1 LLM call** when triage refuses or short-circuits out-of-corpus.
- **2 LLM calls** on the happy path (triage + specialist); **no** second LLM for “critic” — programmatic grounding instead.
- **Conditional rerank** only when dense/BM25 scores are ambiguous (saves GPU + latency).
- **Local** embeddings + numpy shard index (no embedding API spend; fast cold-ish start vs heavyweight vector DB for this scale).

---

## 6. Calibration pass (sample CSV) — what changed and the honesty clause

`sample_support_tickets.csv` is **labelled** and safe for tuning **without** opening the held-out test labels.

**Issues found from traces / eval:**

- HackerRank **test variants** ticket: model escalated despite corpus containing advantages and limitations → **prompt tightening** in `prompts/specialist_hackerrank.md` (“pros/cons / best practice” must **reply** when the doc states tradeoffs, not escalate on wording).
- **Product_area** mismatches: mapped specific Claude **conversation-management** article IDs to **`privacy`** where the rubric expects privacy for private-data / memory / incognito threads; **empty** `product_area` for **escalated + company None** to match outage-style rows; **`conversation_management`** for **out-of-corpus invalid** canned reply (sample’s off-topic row).

**Brutal truth:** These adjustments are **not** “free intelligence.” They encode **observed label patterns** on a **tiny** public sample. They can improve **sample** metrics while carrying **some** risk of mismatch on hidden rows if the test distribution differs. The mitigations are: rules are **narrow** (path prefixes, triage states), **no** test-set peeking, and **critic** still blocks invented contacts/URLs.

**After tuning:** sample eval reached **100%** exact match on status, request_type, and product_area for all 10 sample rows (verify with `python code/main.py eval`). That is **not** a guarantee on the full `support_tickets.csv` leaderboard.

---

## 7. Environment and reproducibility

- **Keys:** `ANTHROPIC_API_KEY` (primary), optional `GEMINI_API_KEY`. Copy `code/.env.example` to repo root or `code/` `.env`.
- **Dotenv loading:** `load_dotenv(REPO_ROOT/.env)`, `load_dotenv(code/.env)`, then cwd — so running from different folders still finds keys.
- **Startup line:** Batch runs print `LLM backend=...` and whether keys are **loaded** (boolean only, not secret values).

---

## 8. Submission checklist (three uploads)

1. **Code zip** — contents of `code/` only, excluding `.venv`, `runs/`, caches, `.env`, `__pycache__`, etc. (Hackathon also expects: no `data/`, no ticket CSVs **inside** the zip.)
2. **Predictions CSV** — `support_tickets/output.csv` from `python code/main.py run` on full `support_tickets.csv`.
3. **Chat transcript** — `%USERPROFILE%\hackerrank_orchestrate\log.txt` (see `AGENTS.md`).

**Entry point:** `python code/main.py run` (from repo with deps installed and index built — first run may build `data/index` artifacts locally).

---

## 9. Commands reference

```text
python code/main.py eval          # labelled sample + eval_report.md
python code/main.py run           # full support_tickets.csv → output.csv
python code/main.py rebuild-index # force rebuild corpus index
```

---

## 10. Honest assessment: will this “win”?

**Strengths**

- **Grounding story is real:** retrieval + citations + programmatic critic + escalation path.
- **Architecture is interview-friendly:** graph, spans, explicit policy.
- **Operational maturity:** rate limits, retries, structured I/O, run traces.
- **Pragmatic provider pivot** with documented rationale.

**Weaknesses / risks**

- **Held-out performance unknown** — sample-perfect tuning can **overfit** rubric quirks.
- **Product_area mapping** is partly **convention**, not semantic truth; judges may weight it differently.
- **Escalation rate** on full run (~13/29 escalated in last batch) may be high or low vs organizer expectations — only they know.
- **Latency + cost** on 29×2 LLM calls is non-trivial; Haiku/Sonnet choices balance that but are not “free.”
- **Interview risk:** If asked “did you tune to the sample?”, answer **truthfully**: yes, using **only** the public labelled set, for prompt and **deterministic** routing/product-area rules — not the hidden rows.

**Bottom line:** The submission is **credible and defensible** as a **production-style** triage harness. Whether it **wins** depends on **hidden labels**, **judge rubric weighting**, and **AI interview** depth — not on local sample scores alone.

---

## 11. Session log (where the “chat transcript” lives)

Per `AGENTS.md`, conversational logging is **outside the repo**:

- Windows: `%USERPROFILE%\hackerrank_orchestrate\log.txt`

That file is append-only and shared across tools; do not commit it.

---

*Generated as a durable handoff for the participant. Last full batch: 29 tickets, Anthropic backend, 0 pipeline errors in terminal summary.*
