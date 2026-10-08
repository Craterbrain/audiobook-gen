"""Shared plumbing for the worker scripts. Import this first: it keeps the real stdout for the protocol and sends
everything libraries print while loading to stderr."""
import json
import os
import sys

_out = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr


def serve(sr: int, handle):
    _out.write(json.dumps({"ready": True, "sr": sr}) + "\n")
    _out.flush()
    for line in sys.stdin:
        req = json.loads(line)
        try:
            handle(req)
            _out.write(json.dumps({"ok": True}) + "\n")
        except Exception as e:   # report, keep serving
            _out.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        _out.flush()
