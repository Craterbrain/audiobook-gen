"""Wait for your reply on the ntfy topic and print it. Run in the background after sending a question there.
Usage: python tools/ntfy_wait.py [--since EPOCH_SECONDS] [--timeout SECONDS]    (default: replies from now on, for an hour)
Exit code 0 = a reply was printed, 2 = none arrived."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audiobook_gen import jobqueue  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--since", type=float, default=0); ap.add_argument("--timeout", type=float, default=3600)
a = ap.parse_args()
reply = jobqueue.wait_for_reply(a.since or time.time(), a.timeout)
print(reply if reply is not None else "no reply")
sys.exit(0 if reply is not None else 2)
