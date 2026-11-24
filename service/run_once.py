"""Entry point for the scheduled GitHub Action: drain the grading queue once, then exit.
Prints only counts, never student data (Action logs of a public repo are public)."""
import logging
import os
import httpx
from .store import Store
from .worker import drain

logging.basicConfig(level=logging.WARNING)


def start_next_run(repo: str, token: str) -> bool:
    """A long paper paused part-way (rate limits or the run's time limit): start another run now instead of waiting
    hours for GitHub's schedule. The Action's own GITHUB_TOKEN may start a workflow_dispatch run."""
    if not repo or not token:
        return False
    try:
        r = httpx.post(f"https://api.github.com/repos/{repo}/actions/workflows/grade.yml/dispatches",
                       headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                "X-GitHub-Api-Version": "2022-11-28"}, json={"ref": "main"}, timeout=20)
        return r.status_code == 204
    except Exception:
        return False


def should_continue(store) -> bool:
    """Only when a job paused AND this run read at least one new page: a provider that refuses everything (a daily
    cap) then stops the chain instead of starting runs forever; the schedule tries again later."""
    return bool(getattr(store, "paused", False)) and getattr(store, "cache_writes", 0) > 0


if __name__ == "__main__":
    store = Store()
    if os.getenv("REQUEUE", "").lower() == "true":
        store.requeue_all()
        print("requeued jobs that needed a teacher or failed")
    print(f"graded/handled {drain(store, limit=25)} job(s)")
    if should_continue(store):
        ok = start_next_run(os.getenv("GITHUB_REPOSITORY", ""), os.getenv("GITHUB_TOKEN", ""))
        print("a long paper paused part-way: " + ("started the next run" if ok else "could not start the next run"))
