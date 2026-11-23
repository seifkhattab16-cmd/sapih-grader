"""Roadmap 3.6: re-test the grader against the teacher's own verdicts already saved in the portal.

    python -m service.verdict_eval [--limit 30] [--kind submission|quiz] [--no-guidance] [--out report.json]

Takes approved AI drafts the teacher checked herself (never "Approve all", never an unmarked question), grades the
same work again with the CURRENT prompts and models, and compares each question with her mark, per work type
(working / proof / sketch). Use it before and after a prompt change. Nothing is written to Supabase: no draft, no
job, no grade. Needs the same environment as the grader (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, the AI keys).
The printed output holds counts and ids only; --out writes marks per question (keep that file off GitHub)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from . import grader_core
from .eval_harness import summarise, TYPES
from .materials import fetch_bytes
from .providers import complete as default_complete
from .store import Store
from .worker import assignment_for, guidance_for



def checked_questions(draft: dict) -> dict:
    """{ref: (teacher marks, work type)} for the questions she marked herself."""
    out = {}
    for q in draft.get("questions") or []:
        t = (q or {}).get("teacher") or {}
        if not t or t.get("bulk") or t.get("unmarked"):
            continue
        try:
            mark = float(t.get("awarded"))
        except (TypeError, ValueError):
            continue
        typ = t.get("type") or q.get("work_type") or "working"
        out[str(q.get("ref"))] = (mark, typ if typ in TYPES else "working")
    return out


def compare(draft: dict, out) -> list[dict]:
    """One result per work type in the draft, shaped for eval_harness.summarise."""
    got = {grader_core._lead(q["ref"]): q for q in (out.questions if out else [])}
    by: dict[str, dict] = {}
    for ref, (mark, typ) in checked_questions(draft).items():
        q = got.get(grader_core._lead(ref))
        r = by.setdefault(typ, {"id": draft["id"], "type": typ, "status": out.status if out else "error",
                                "flags": list(out.flags) if out else ["error"], "via": out.via if out else "", "rows": []})
        r["rows"].append({"ref": ref, "teacher": mark, "ai": None if q is None else q["awarded_marks"],
                          "max": None if q is None else q["max_marks"],
                          "before": next((float(x.get("awarded_marks") or 0) for x in draft["questions"]
                                          if str(x.get("ref")) == ref), None)})
    return list(by.values())


def load_drafts(store: Store, kind: str | None, limit: int) -> list[dict]:
    params = {"status": "eq.approved", "select": "id,kind,submission_id,questions,approved_at",
              "order": "approved_at.desc", "limit": str(limit * 3)}
    if kind:
        params["kind"] = f"eq.{kind}"
    r = store.c.get(store._u("ai_grades"), headers=store._h(), params=params)
    r.raise_for_status()
    return [d for d in r.json() or [] if checked_questions(d)][:limit]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--kind", choices=("submission", "quiz", "class_question"))
    ap.add_argument("--no-guidance", action="store_true", help="grade without the teacher's notes and corrections")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    store = Store()
    drafts = load_drafts(store, a.kind, a.limit)
    if not drafts:
        print("No approved drafts with the teacher's own marks yet. Mark some drafts in the portal first.")
        return 0
    results = []
    for d in drafts:
        out = None
        try:
            sub = store.get_submission(d["kind"], d["submission_id"])
            assignment, why = assignment_for(store, d["kind"], sub) if sub else (None, "gone")
            if assignment:
                job = {"kind": d["kind"], "submission_id": d["submission_id"]}
                g = None if a.no_guidance else guidance_for(store, job, sub)
                out = grader_core.grade_submission(assignment, sub.get("data") or {}, fetch_bytes, default_complete,
                                                   guidance=g)
        except Exception as e:                     # one bad paper must not stop the run
            print(f"{d['id'][:8]}  error {type(e).__name__}")
        rs = compare(d, out)
        results.extend(rs)
        for r in rs:
            print(f"{d['id'][:8]}  {r['type']:<8}{r['status']:<14}"
                  + " ".join(f"{x['ref']}:{x['teacher']:g}/{'-' if x['ai'] is None else x['ai']}" for x in r["rows"]))
    summary = summarise(results)
    before = summarise([{**r, "rows": [{**x, "ai": x["before"]} for x in r["rows"]]} for r in results])
    print("\nteacher/AI per question above. Agreement with the teacher (exact marks):")
    print("type      questions  before%  now%   within1%  bar(80/95)")
    for t in TYPES:
        if t in summary:
            s, b = summary[t], before.get(t, {})
            print(f"{t:<10}{s['questions']:<11}{b.get('exact_pct', 0):<9}{s['exact_pct']:<7}{s['within1_pct']:<10}"
                  f"{'PASS' if s['passes_bar'] else 'assisted-only'}")
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "before": before, "results": results}, indent=2),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
