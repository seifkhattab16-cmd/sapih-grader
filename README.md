# saiph-grader

Grades student submissions for the Saiph classroom portal. Runs as a scheduled GitHub Action (every ~5 min): claims queued
jobs from Supabase, reads the student's work and the mark scheme, and stores a **draft** grade for the teacher to approve.
No student data is stored in this repo. Secrets (Supabase secret key, AI keys) live in GitHub Actions secrets.
Details: `service/README.md`. Tests: `pip install -r service/requirements.txt && python -m pytest service/tests -q`.
