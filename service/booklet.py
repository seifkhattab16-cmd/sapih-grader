"""Marked-work R1.2: a generated booklet is marked against each question's own mark scheme from the question bank,
not against a scanned model answer whose pages have to be matched by reading question numbers.

The booklet's question paper lists its bank questions and pages (materials.booklet_manifest). The bank stores, per
question, a mark scheme as one line per mark ({part, markType, answer, notes}) and a model answer per part
({part, marks, answer, working}). This module turns those into the text the marker reads."""
from __future__ import annotations

MAX_STEM = 1500
MAX_SCHEME = 3500
MAX_ANSWER = 3500


def _s(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def scheme_lines(mark_scheme) -> str:
    """'(a) M1: answer (notes)' per mark, in the bank's order. Plain text kept as it is."""
    if isinstance(mark_scheme, str):
        return mark_scheme.strip()
    out = []
    for m in mark_scheme or []:
        if not isinstance(m, dict):
            continue
        part, mt, ans, notes = _s(m.get("part")), _s(m.get("markType")), _s(m.get("answer")), _s(m.get("notes"))
        if not (ans or notes):
            continue
        head = " ".join(x for x in (part, mt) if x)
        out.append(f"{head}: {ans}" + (f" ({notes})" if notes else "") if head else ans + (f" ({notes})" if notes else ""))
    return "\n".join(out)


def answer_lines(model_answer) -> str:
    """'(a) [M1 A1] answer' and its working, per part."""
    if isinstance(model_answer, str):
        return model_answer.strip()
    out = []
    for m in model_answer or []:
        if not isinstance(m, dict):
            continue
        part, marks, ans, work = _s(m.get("part")), _s(m.get("marks")), _s(m.get("answer")), _s(m.get("working"))
        if not (ans or work):
            continue
        out.append(" ".join(x for x in (part, f"[{marks}]" if marks else "") if x) + (f" Answer: {ans}" if ans else ""))
        if work:
            out.append(work)
    return "\n".join(out)


def booklet_references(manifest: dict | None, rows: dict[str, dict]) -> dict[str, dict]:
    """Printed question number (as text, "4") -> {"id", "marks", "ms", "ma"}: the marker's reference for that question.
    A question whose bank row has neither a mark scheme nor a model answer gets empty texts (listed for the teacher,
    never marked against nothing)."""
    out: dict[str, dict] = {}
    for q in (manifest or {}).get("q", []):
        row = rows.get(q.get("id")) or {}
        n = str(q.get("n"))
        marks = int(row.get("marks") or q.get("marks") or 0)
        stem = _s(row.get("stem"))[:MAX_STEM]
        ms, ma = scheme_lines(row.get("mark_scheme"))[:MAX_SCHEME], answer_lines(row.get("model_answer"))[:MAX_ANSWER]
        head = f"[{marks} marks]" + (f" {stem}" if stem else "")
        # The question text goes once, with the mark scheme (or the model answer when there is no scheme).
        out[n] = {"id": q.get("id"), "marks": marks,
                  "ms": f"{head}\nMark scheme:\n{ms}" if ms else "",
                  "ma": (f"Model answer:\n{ma}" if ms else f"{head}\nModel answer:\n{ma}") if ma else ""}
    return out
