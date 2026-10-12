"""Things done to a book that both the GUI and the Ask Claude helper need, with no Gradio in them. Problems raise ValueError (the GUI turns
those into its own error messages)."""
import json
from pathlib import Path

from . import prep, workflow


def making_now(work) -> dict | None:
    """The running job if the queue is making this very book, else None. Edits to a book's lines, chapters or music wait for it."""
    from . import jobqueue
    run = jobqueue.now_running()
    return run if run and Path(run["work"]).resolve() == Path(work).resolve() else None


def refuse_while_making(work, what: str = "change this") -> None:
    run = making_now(work)
    if run:
        raise ValueError(f"“{run['title']}” is being made right now, so you can't {what} until it finishes. "
                         "Pause or cancel it on the Queue tab first (clips already made are kept).")


def enqueue(work, title: str, author: str, cover: str, chapters, not_before: str, window: str, send_to: str, supersede=None):
    """Queue a book once. If it already has a waiting job, that job is updated instead of adding a second one. Returns (job, updated).
    Refuses a book whose text has not been prepared."""
    from . import jobqueue
    work = Path(work)
    st = prep.status(work)
    if not (work / "segments.json").exists() or not st["ready"]:
        raise ValueError(f"The text has not been prepared ({st['why']}). Open the Book tab, tick the preparation steps and press Prepare.")
    waiting = jobqueue.pending_for(str(work))
    if waiting:
        jobqueue.update(waiting["id"], title=title, author=author, cover=cover, chapters=chapters or None, not_before=not_before,
                        window=window, send_to=send_to, status="queued", note="")
        job, updated = {**waiting, "title": title}, True
    else:
        job, updated = jobqueue.add(str(work), title, author, cover, "", "", chapters, not_before, window, send_to), False
        jobqueue.ensure_supervisor()
    if supersede:
        jobqueue.supersede(supersede, job["id"])
    workflow.stamp_queued(work, job["id"], title, author)          # what this build is made from, so a later change can say what it costs
    return job, updated


def rebuild(work, send: bool = False, device: str = "") -> tuple[dict, bool]:
    """Queue the book again with what its last run used (chapters, window, title). Clips that exist are reused. Sends to a phone only if asked."""
    from . import jobqueue
    work = Path(work)
    if send and not device:
        raise ValueError("Choose which phone to send it to (or do not send).")
    book = json.loads((work / "chapters.json").read_text())
    last = next(iter(jobqueue.jobs_for(str(work))), None) or {}
    cover = last.get("cover") or next((str(work / n) for n in ("cover_custom.jpg", "cover.jpg") if (work / n).exists()), "")
    return enqueue(work, last.get("title") or book.get("title", work.name), last.get("author") or book.get("author", ""), cover,
                   last.get("chapters"), "", last.get("window", ""), device if send else "")


def reapply_extras(work) -> None:
    """The spoken opening and section titles the music plan asks for (preparing again drops them), and keep the book 'prepared'."""
    from . import music, titles
    work = Path(work)
    if not (work / "segments.json").exists():
        return
    was_ready = prep.status(work)["ready"]
    plan = music.load_plan(work)
    intro = plan["intro"]
    layered = bool(plan["enabled"] and intro.get("track") and intro.get("title"))
    music.add_opening(work, intro["title"] if layered else "", intro.get("title_pause", 0.0))
    t = plan.get("titles") or {}
    if t.get("speak"):
        titles.apply(work, True, float(t.get("pause", 2.8)))
    if was_ready:
        prep.restamp(work)
