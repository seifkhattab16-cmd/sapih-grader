"""Entry point for the scheduled GitHub Action: drain the grading queue once, then exit.
Prints only counts, never student data (Action logs of a public repo are public)."""
import logging
from .store import Store
from .worker import drain

logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    print(f"graded/handled {drain(Store(), limit=25)} job(s)")
