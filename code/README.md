# Application: grounded triage CLI

Implementation package for the **multi-domain corpus-grounded support triage** system. The **portfolio overview, architecture diagram, and ops guidance** live in the [repository root `README.md`](../README.md).

---

## Entry point

```bash
python main.py run          # support_tickets.csv → output.csv
python main.py eval        # labelled sample → metrics + eval_report.md
python main.py rebuild-index
python main.py trace --run latest
```

Run from repository root (`python code/main.py …`) or from `code/` with `PYTHONPATH` set as today.

---

## LLM budget (harness design)

- **0 calls:** fast path (trivial chat, injections, empty ticket, …).
- **1 call:** triage-only terminal paths (e.g. out-of-corpus canned reply after triage).
- **2 calls:** standard path: **Haiku** triage + **Sonnet** specialist (defaults).

Programmatic **critic**—no extra LLM.

---

## Layout

```text
code/
├── main.py
├── requirements.txt
├── .env.example
├── hro/
│   ├── config.py
│   ├── schemas.py
│   ├── graph.py
│   ├── runner.py
│   ├── eval.py
│   ├── observability.py
│   ├── safety_rules.py
│   ├── corpus/          # loader, chunker, product_area_map.json
│   ├── index/           # build, store, bm25, retriever, device
│   ├── llm/             # client (Anthropic / Gemini), rate_limiter, prompts
│   └── agents/          # fast_path, triage, specialists, critic, composer
├── prompts/
├── tests/
└── runs/                # gitignored — trace.jsonl + summary per batch
```

---

## Tuning

Thresholds and model names: `hro/config.py`.  
Prompts: `prompts/*.md`, `prompts/few_shot.json`.

---

## Secrets

Copy `.env.example` to the **repo root** or **`code/`** and set `ANTHROPIC_API_KEY`. Optional: `GEMINI_API_KEY` and `HRO_BACKEND=gemini`.

---

## Tests

```bash
pytest tests/ -q
```

Some tests are skipped without optional heavy deps or API access; CI-friendly unit tests live under `tests/test_*.py`.
