# Evaluation pages for step 3.5

30 pages of your own handwriting, marked by you. The photos never leave your PC (`eval/pages/` is git-ignored).

## What to make
| Type | How many | What to write |
|---|---|---|
| `working` | 10 | Normal solutions: a few questions per page with working. Make some answers right, some with a wrong step, some blank, some with a right answer but no method. |
| `proof` | 10 | Short proofs (e.g. prove an identity, a sum, a factor). Include correct ones, ones with a missing step, and ones with a wrong step. |
| `sketch` | 10 | Graphs: curve with intercepts, asymptotes, shape and labels. Some complete, some missing a feature. |

Photograph each page flat and bright (like a student would). Save as PNG/JPG in `eval/pages/`, named `w01.jpg`, `p01.jpg`, `s01.jpg`...
One file per item (a PDF is fine for several pages).

## manifest.json
Copy `manifest.example.json` to `manifest.json` and add one entry per file:

- `id`: the file name without extension.
- `type`: `working`, `proof` or `sketch`.
- `page`: `pages/w01.jpg`.
- `scheme`: the mark scheme as plain text with the marks per question (M1 / A1 / B1), e.g. `Q1 (3 marks): M1 differentiates, A1 12x^2, A1 correct answer`.
- `model`: optional model answer text.
- `teacher_marks`: the marks **you** gave per question, e.g. `{"1": 3, "2a": 1}`. Mark before looking at what the AI says.

## Run
```
cd saiph-grader
$env:PYTHONIOENCODING="utf-8"
python -m service.eval_harness eval/manifest.json --out eval/report.json
```
Needs the same AI provider keys as the grader in your environment (see `service/README.md`). Nothing is written to Supabase.

The report prints exact match and within-1-mark agreement per type. Pass bar: exact on at least 80% of questions and within 1 mark on at least 95%; otherwise that type stays assisted-only.
