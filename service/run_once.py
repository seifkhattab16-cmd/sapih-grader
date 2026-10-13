"""Entry point for the scheduled GitHub Action: drain the grading queue once, then exit.
Prints only counts, never student data (Action logs of a public repo are public)."""
import logging
import os
from .store import Store
from .worker import drain

logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    if os.getenv("REQUEUE", "").lower() == "true":
        Store().requeue_all()
        print("requeued jobs that needed a teacher or failed")
    print(f"graded/handled {drain(Store(), limit=25)} job(s)")
