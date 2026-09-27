"""Run sample_data/test_questions.json through the real pipeline and print a report.

    python -m scripts.seed_demo            # once
    python -m scripts.run_test_questions [--delay SECONDS] [--only ID,ID]   # needs an LLM key in .env

--delay pauses between questions to stay under free-tier per-minute rate limits.

Writes the full JSON results (answers, references, traces) to data/test_results.json.
"""
import argparse
import asyncio
import json
from pathlib import Path

from app.config import settings
from app.db import get_conn, init_db
from app.pipeline import answer_question

QUESTIONS = Path(__file__).resolve().parent.parent / "sample_data" / "test_questions.json"


async def main(delay: float, only: set[str]) -> None:
    init_db()
    with get_conn() as conn:
        orgs = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM organizations")}
    results = []
    cases = [c for c in json.loads(QUESTIONS.read_text(encoding="utf-8")) if not only or c["id"] in only]
    for i, case in enumerate(cases):
        if i and delay:
            await asyncio.sleep(delay)
        org_id = orgs.get(case["org"])
        if not org_id:
            raise SystemExit(f"Organization '{case['org']}' not found - run python -m scripts.seed_demo first")
        out = await answer_question(org_id, case["org"], case["question"])
        t = out["trace"]
        overall = t.get("combined", {}).get("overall", "-")
        refs = ", ".join(f"{r['document']} p{r['page']}" for r in out["references"]) or "-"
        print(f"\n[{case['id']}] {case['category']}\n  Q: {case['question']}\n"
              f"  type={t.get('classification', {}).get('label')}  evidence={t.get('evidence', {}).get('verdict')}  "
              f"decisions={len(t.get('decisions', []))}  overall={overall}  status={out['status']}  "
              f"llm_calls={t['llm_calls']}  {out['latency_ms']}ms\n"
              f"  expected: {case['expected']}\n  answer: {out['answer'][:400]}\n  sources: {refs}")
        results.append({**case, "result": out})
    path = settings.data_dir / "test_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--only", default="", help="comma-separated question ids, e.g. NB-4,AC-2")
    args = parser.parse_args()
    asyncio.run(main(args.delay, {x.strip() for x in args.only.split(",") if x.strip()}))
