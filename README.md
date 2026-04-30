# CiteAudit

Detect hallucinated / fake academic citations using URL probing, Google Search,
Google Scholar, and an LLM judge.

## Layout

```
citeaudit/
├── config.py              # reads API keys from env vars
├── pdf_processor.py
├── serp_verify.py         # core verification pipeline
├── test_serp.py           # CLI runner (concurrent, resumable, fail-fast)
├── requirements.txt
└── data/
    └── benchmark.json     # 9442 citations with ground-truth labels
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."         # OpenAI (fallback judge)
export GEMINI_API_KEY="..."            # Google Gemini (primary judge)
export SERPAPI_API_KEY="..."           # SerpApi (Google + Google Scholar)
# optional
export GEMINI_MODEL="gemini-3-flash-preview"
```

You need accounts on Google AI Studio (Gemini), OpenAI, and SerpApi.
Running the full benchmark uses roughly 8000–9000 SerpApi calls.

## Usage

```bash
# Run the full benchmark
python3 test_serp.py \
    --file data/benchmark.json \
    --gt all --start 0 --limit 9442 \
    --workers 16 \
    --output results/full_run.json
```

Useful flags:

| Flag | Description |
|---|---|
| `--workers N` | Concurrent worker threads (default 5) |
| `--gt all\|true\|false` | Filter by GT label |
| `--start N --limit M` | Verify a slice of the dataset |
| `--no-resume` | Force fresh run; ignore any existing output |

The runner is **resumable** (re-running the same command picks up where it left
off) and **fail-fast** (auth/quota errors abort with exit 2; safe to resume).

## Benchmark format

Each record in `data/benchmark.json` (under the `data` key):

```json
{
  "source_type": "realworld" | "generated",
  "citation": "<full citation string>",
  "GT": true | false,
  "Predict": true | false
}
```

- `realworld` (3356) — citations extracted from real PDFs.
- `generated` (6086) — LLM-produced citations, real or fabricated.
- `GT` — ground-truth label (`true` = real, `false` = fake).

## Output format

```json
{
  "results": [
    {
      "index": 0,
      "source_type": "realworld",
      "citation": "...",
      "GT": true,
      "Pred": true,
      "Correct": true,
      "verification": { "method": "...", "found": true, "note": "...", ... }
    }
  ],
  "summary": { "total": 9442, "correct": 8616, "accuracy": 0.9125 }
}
```

## Evaluation

Treat "fake citation" as the positive class:

- **TP** — actually fake AND flagged as fake
- **FN** — actually fake but flagged as real (missed)
- **TN** — actually real AND flagged as real
- **FP** — actually real but flagged as fake (over-flagged)

`Accuracy = (TP+TN)/N`, `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`,
`F1 = 2·P·R/(P+R)`.
