#!/usr/bin/env python3
"""Test SerpApi Google Search + Google Scholar verification on result.json.

Features:
- Concurrent workers (--workers N)
- Resume: if --output already exists, completed indices are skipped
- Fail-fast: aborts the run when Gemini or SerpAPI report a fatal error
"""
import argparse
import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path

import openai
from serp_verify import (
    FatalAPIError,
    GeminiFatalError,
    SerpAPIFatalError,
    initialize_provider_routing,
    verify_with_serp,
)
from config import OPENAI_API_KEY


def load_samples(path: Path, limit: int, start: int, gt_filter: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("data", data)
    if not isinstance(rows, list):
        raise ValueError("result.json must contain a list or a top-level data list")

    if gt_filter != "all":
        wanted = gt_filter == "true"
        rows = [row for row in rows if row.get("GT") is wanted]

    return rows[start:start + limit]


def write_results(path: Path, payload: dict):
    """Atomic write with fsync — survives kill / power-loss without corrupting the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_existing(path: Path) -> dict:
    """Load previously written results if any. Returns {index: result_entry}."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {r["index"]: r for r in data.get("results", []) if "index" in r}
    except Exception as exc:
        print(f"[resume] failed to read {path}: {exc} — starting fresh")
        return {}


def process_one(offset: int, row: dict, client: openai.OpenAI,
                abort_event: threading.Event) -> dict:
    if abort_event.is_set():
        raise RuntimeError("aborted before start")

    citation_text = row.get("citation", "")
    gt = bool(row.get("GT"))
    ref = {"index": offset, "raw": citation_text}

    print(f"[idx {offset}] start | GT={gt} | {citation_text[:80]}")
    result = verify_with_serp(ref, client)
    pred = bool(result.get("found", False))
    print(f"[idx {offset}] done  | Pred={pred} | Correct={pred == gt} | method={result.get('method', 'N/A')}")

    return {
        "index": offset,
        "source_type": row.get("source_type"),
        "citation": citation_text,
        "GT": gt,
        "original_predict": row.get("Predict"),
        "Pred": pred,
        "Correct": pred == gt,
        "verification": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Run SerpApi citation verification on result.json samples.")
    parser.add_argument("--file", default="result.json", help="Path to result.json")
    parser.add_argument("--start", type=int, default=0, help="Start index after optional GT filtering")
    parser.add_argument("--limit", type=int, default=3, help="Number of citations to verify")
    parser.add_argument("--gt", choices=["all", "true", "false"], default="all", help="Filter by GT label")
    parser.add_argument("--output", default="serp_results.json", help="Where to write verification results")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker threads")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output file and start fresh")
    args = parser.parse_args()

    initialize_provider_routing()

    samples = load_samples(Path(args.file), args.limit, args.start, args.gt)
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    output_path = Path(args.output)

    # Resume: load any prior results from same output file
    completed = {} if args.no_resume else load_existing(output_path)
    if completed:
        print(f"[resume] {len(completed)} previously verified entries loaded from {output_path}")

    pending = [(offset, row) for offset, row in enumerate(samples, start=args.start)
               if offset not in completed]
    if not pending:
        print(f"[resume] all {len(samples)} samples already verified — nothing to do")
        return

    if completed:
        print(f"[resume] resuming with {len(pending)} pending / {len(samples)} total")

    output = {
        "source_file": str(Path(args.file).resolve()),
        "start": args.start,
        "limit": args.limit,
        "gt_filter": args.gt,
        "workers": args.workers,
        "results": [],
        "summary": {"total": 0, "correct": 0, "accuracy": None},
    }

    results_by_index = dict(completed)
    write_lock = threading.Lock()
    abort_event = threading.Event()
    fatal_msg: list = []
    done = len(completed)
    total = len(samples)

    def commit(entry: dict):
        nonlocal done
        with write_lock:
            results_by_index[entry["index"]] = entry
            sorted_results = [results_by_index[i] for i in sorted(results_by_index)]
            correct = sum(1 for r in sorted_results if r["Correct"])
            done = len(sorted_results)
            output["results"] = sorted_results
            output["summary"] = {
                "total": done,
                "correct": correct,
                "accuracy": correct / done if done else None,
            }
            write_results(output_path, output)
            print(f"  >> progress {done}/{total} | accuracy={correct}/{done} ({correct/done:.2%})")

    def trigger_abort(source: str, exc: Exception):
        with write_lock:
            if not fatal_msg:
                fatal_msg.append(f"{source}: {exc}")
        abort_event.set()

    # Persist completed results first so resuming + writing summary is consistent
    if completed:
        commit_seed = next(iter(completed.values()))
        commit(commit_seed)  # triggers write with all completed in sorted order

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, offset, row, client, abort_event): offset
            for offset, row in pending
        }
        try:
            for fut in concurrent.futures.as_completed(futures):
                if abort_event.is_set():
                    break
                try:
                    entry = fut.result()
                    if entry:
                        commit(entry)
                except GeminiFatalError as exc:
                    print(f"\n[FATAL] Gemini API: {exc}")
                    trigger_abort("Gemini", exc)
                except SerpAPIFatalError as exc:
                    print(f"\n[FATAL] SerpAPI: {exc}")
                    trigger_abort("SerpAPI", exc)
                except FatalAPIError as exc:
                    print(f"\n[FATAL] API: {exc}")
                    trigger_abort("API", exc)
                except RuntimeError as exc:
                    if "aborted" not in str(exc):
                        print(f"  worker error: {exc}")
                except Exception as exc:
                    print(f"  worker error: {exc}")
        finally:
            if abort_event.is_set():
                # Cancel pending futures so executor shuts down quickly
                for f in futures:
                    f.cancel()

    if abort_event.is_set():
        print("\n" + "=" * 80)
        print(f"ABORTED: {fatal_msg[0] if fatal_msg else 'unknown fatal error'}")
        print(f"Verified so far: {done}/{total}")
        print(f"Partial results saved to: {output_path.resolve()}")
        print("Re-run the same command to resume from the last completed entry.")
        sys.exit(2)

    correct = output["summary"]["correct"]
    print("\n" + "=" * 80)
    print(f"Accuracy on this run: {correct}/{done} ({correct / done:.2%})" if done else "No samples")
    print(f"Output written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
