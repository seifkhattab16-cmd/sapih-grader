# Grading host (roadmap 3.2)

Small FastAPI service that grades student work automatically and stores a **draft** for the teacher to approve (3.3).

Flow: a student submits -> trigger (`supabase_grading.sql` in the portal repo) queues a `grading_jobs` row ->
this host claims it, reads the student's work (typed text, photo or PDF) and the assignment's marking scheme /
model answer, asks an AI provider, and writes `ai_grades` (draft). It never writes `submissions.grade`.

Not graded (job goes to `needs_teacher`): work it cannot read, or an assignment with no readable mark scheme.

## Secrets (host settings, never in git)
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- `GRADER_WEBHOOK_SECRET` (any long random string; the keep-warm ping sends it as header `x-grader-secret`)
- at least one of `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` (optional `AI_ROUTE_ORDER`, model overrides as in the portal's `ai` function)

## Run / test
```
pip install -r service/requirements.txt
python -m pytest service/tests -q
uvicorn service.app:app --port 8080
docker build -f service/Dockerfile -t grader .
```
`POST /tick` (with the secret) drains the queue; `GET /health` is open. A GitHub Action should call `/tick` every ~5 minutes to keep a free host awake.
