"""Step 3.5 B: measure the grader against the teacher's own marks.

    python -m service.eval_harness eval/manifest.json [--out eval/report.json]

Reads a manifest of hand-written pages (see eval/README.md), runs the real grader on each one with the
providers' keys from the environment, and prints agreement per work type (working / proof / sketch).
Nothing is written to Supabase. Photos stay on this machine (eval/pages is git-ignored)."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .grader_core import grade_submission, _lead
from .providers import complete as default_complete

TYPES = ("working", "proof", "sketch")
FAKE = "https://eval.local/"


def load_manifest(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("items", [])
    for it in items:
        if it.get("type") not in TYPES:
            raise ValueError(f"{it.get('id')}: type must be one of {TYPES}")
        for k in ("id", "page", "scheme", "teacher_marks"):
            if k not in it:
                raise ValueError(f"{it.get('id')}: missing '{k}'")
    return items


def run_item(it: dict, base: Path, complete: Callable) -> dict:
    page = (base / it["page"]).resolve()
    files = {FAKE + "work/" + page.name: page.read_bytes()}
    assignment = {"title": it["id"], "markingSchemeNotes": it["scheme"], "modelAnswerNotes": it.get("model", "")}
    submission = {"fileUrl": FAKE + "work/" + page.name}
    out = grade_submission(assignment, submission, lambda url: files[url], complete=complete)
    got = {_lead_ref(q["ref"]): q for q in out.questions}
    rows = []
    for ref, want in it["teacher_marks"].items():
        q = got.get(_lead_ref(ref))
        rows.append({"ref": ref, "teacher": want, "ai": None if q is None else q["awarded_marks"],
                     "max": None if q is None else q["max_marks"]})
    return {"id": it["id"], "type": it["type"], "status": out.status, "flags": out.flags, "via": out.via, "rows": rows}


def _lead_ref(ref: str) -> str:
    return str(ref).strip().lower()


def summarise(results: list[dict]) -> dict:
    by = defaultdict(lambda: {"questions": 0, "exact": 0, "within1": 0, "over": 0, "under": 0,
                              "ungraded_items": 0, "items": 0, "flagged_items": 0})
    for r in results:
        s = by[r["type"]]
        s["items"] += 1
        if r["status"] != "graded":
            s["ungraded_items"] += 1
        if r["flags"]:
            s["flagged_items"] += 1
        for row in r["rows"]:
            s["questions"] += 1
            if row["ai"] is None:                       # question missing from the AI's answer counts as a miss
                s["under"] += 1
                continue
            d = row["ai"] - row["teacher"]
            s["exact"] += d == 0
            s["within1"] += abs(d) <= 1
            s["over"] += d > 0
            s["under"] += d < 0
    out = {}
    for t, s in by.items():
        n = max(1, s["questions"])
        out[t] = {**s, "exact_pct": round(100 * s["exact"] / n, 1), "within1_pct": round(100 * s["within1"] / n, 1),
                  "passes_bar": s["exact"] / n >= 0.80 and s["within1"] / n >= 0.95}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out")
    ap.add_argument("--only", choices=TYPES)
    a = ap.parse_args(argv)
    mp = Path(a.manifest)
    items = [i for i in load_manifest(mp) if not a.only or i["type"] == a.only]
    results = []
    for it in items:
        try:
            r = run_item(it, mp.parent, default_complete)
        except Exception as e:                               # one bad page must not stop the run
            r = {"id": it["id"], "type": it["type"], "status": "error", "flags": [str(e)[:120]], "via": "", "rows": [
                {"ref": k, "teacher": v, "ai": None, "max": None} for k, v in it["teacher_marks"].items()]}
        results.append(r)
        print(f"{r['id']:<14}{r['type']:<9}{r['status']:<14}" + " ".join(f"{x['ref']}:{x['teacher']}/{x['ai']}" for x in r["rows"]))
    summary = summarise(results)
    print("\ntype      questions  exact%  within1%  ungraded  flagged  bar(80/95)")
    for t in TYPES:
        if t in summary:
            s = summary[t]
            print(f"{t:<10}{s['questions']:<11}{s['exact_pct']:<8}{s['within1_pct']:<10}{s['ungraded_items']:<10}"
                  f"{s['flagged_items']:<9}{'PASS' if s['passes_bar'] else 'assisted-only'}")
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
