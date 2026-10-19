"""Typed S1 answers against the typed scheme derived from the bank's worked solution (roadmap S4.3).
  python eval/run_s1_typed.py            needs the same AI provider keys as the grader; prints agreement with the teacher-style marks
Nothing is written to Supabase. Blank answers must come back needs_teacher / 0 without an AI call."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from service import grader_core, worker
from service.providers import ProviderError

BANK = pathlib.Path(r"C:\Users\seifk\Downloads\convo with gemi\gnerator\backend\data\all_questions_bank.json")
bank = {q["id"]: q for q in json.loads(BANK.read_text(encoding="utf-8")) if q["id"].startswith("pp_wst01")}
cases = json.loads((pathlib.Path(__file__).parent / "s1_typed_cases.json").read_text(encoding="utf-8"))
exact = within1 = n = 0
for c in cases:
    step = [s for s in bank[c["q"]]["modelAnswer"] if s["part"] == c["part"]][0]
    scheme = worker.scheme_from_model_answer([step])
    maxm = sum(int(x) for x in __import__("re").findall(r"[MABmab]{1,2}(\d)", step["marks"].replace("dM", "M").replace("cso", "").replace("ft", "")))
    assignment = {"title": f"S1 {c['part']}", "description": f"Question part {c['part']} [{maxm} marks]", "markingSchemeNotes": scheme}
    try:
        out = grader_core.grade_submission(assignment, {"text": c["student"]}, lambda u: b"")
    except ProviderError as e:
        sys.exit(f"NOT RUN: {e}. Set GROQ_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY in this shell and run again.")
    got = out.total_awarded if out.status == "graded" else 0
    n += 1; exact += got == c["expected"]; within1 += abs(got - c["expected"]) <= 1
    print(f"{c['id']} expected {c['expected']} got {got} status {out.status} flags {out.flags} | {c['note']}")
print(f"exact {exact}/{n}  within 1 mark {within1}/{n}  (pass bar: exact >= 80%, within 1 >= 95%)")
