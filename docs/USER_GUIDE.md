# Audiobook Gen: user guide

This guide walks through making an audiobook from a book file, start to finish, using the app. For the technical side (settings files, command line, how the pieces work) see the [README](../README.md).

Everything runs on your own computer. Your books, recordings and cloned voices never leave it.

## 1. Starting the app

Run `./launch.sh` (or use the *Audiobook Gen* entry in your app menu). It opens http://127.0.0.1:7860 in your browser. `./stop.sh` stops the app; books being made keep going, because the queue is a separate background service.

Under the title is a **strip** with the six steps and their state:

| Mark | Meaning |
|---|---|
| ✓ | done |
| ! | out of date (you changed something earlier) |
| ● | needed |
| ○ | worth a look |
| ⟳ | running |
| 🔒 | needs an earlier step first |

Every page starts with a banner that says what the step is for and which step should come first. Follow the numbers and the steps will not break each other.

## 2. Making a book, step by step

### ① Book
Upload your book. Supported: `.epub`, `.txt`, `.md`, `.html`, `.docx`, `.pdf` (needs `pdftotext`), and with calibre installed also `.mobi`, `.azw3`, `.fb2`, `.rtf`, `.odt`.

* The app reads the book and **checks that it understood the layout** (a play, scripture with verses, a book with chapters, or plain text). If its first reading looks wrong it tries the others by itself. It only asks you when none of them looks right.
* For some kinds of book (scripture, plays) it offers a **preset** once, in a pop-up. Accept it, or decline and set things yourself.
* Tick the chapters you want, fix the title and author, then press **Prepare**. Preparation splits the text into who speaks which line, builds the pronunciation list, and cleans the text (stray spaces, small-capital splits, footnotes read as "Footnote one…").
* If the same book is already in the queue or finished, you get a warning and a choice: **replace** the earlier one (and optionally delete its work folder) or keep both.

### ② Voices
* **Parse speakers** finds who speaks each line. Choose *Different voice for each character* or *One narrator reads everything*.
* Each character gets an engine and voice, a pace slider and a **▶ Hear** button. Changes save as you make them.
* **Find genders** asks the small language model; correct any row by hand. Genders pick sensible default voices.
* You can move lines between characters, or merge two characters.
* Voices: **Kokoro** presets (fast), or your own **cloned** voices from the Voice lab (Chatterbox is the most expressive, F5 and Qwen3 also work).

### ③ Names
The pronunciation list. Every name or hard word has a spelling hint for the voice.

* **Where the model placed each name.** A small language model reads each name in its sentences and decides whether it belongs to US English, UK English or another language (Spanish, French, German…). A US town with a Spanish name keeps its Spanish sound; an English surname in an American book stays American. Use the **setting filter** to see names by where they were placed, and **re-check** a group if you disagree.
* Edit IPA (for Kokoro) or a respelling (for F5/Chatterbox) directly. What you type is never overwritten.
* **Look up IPA online** fills blanks from Wiktionary and other dictionaries; **Auto-spell** makes respellings.
* Roman numerals in chapter headings are handled as headings and kept out of this list.

### ④ Extras
* **Cover**: search free pictures on Wikimedia Commons or upload your own, pick a layout and colour, press *Use this cover*.
* **Music**: opening song, music between sections ("bridges"), ending. Details in section 4.
* **Spoken titles**: each section's title is read aloud by default; untick to turn it off.

### ⑤ Make
Pacing sliders, the engines panel, how names are spoken (real IPA, respelling, plain). Then choose **start now** or a start time, optionally **only overnight**, optionally **send to my phone** when finished, and press the button. The book goes onto the queue. Before the first run you see the measured speed of your own machine and an estimate of how long it will take.

### ⑥ Fix
Find any clip by the words in it, hear it, change the words or settings, and remake just that clip. After editing the pronunciation list, search the name to see which clips change, then **Rebuild**. Only clips whose words changed are made again. Your fixes survive later re-runs.

## 3. Queue and Done

### Queue
Every book waits here. The runner makes them one at a time and watches each:

1. **Speech**: the clips are made. A progress count and a time estimate are shown.
2. **Clip check**: odd clips (dead air, breakage, wrong pace) are remade with other seeds.
3. **Name check**: a speech recogniser listens to each name and compares it to the intended sound; weak ones are retaken or respelled. The note on the job says what it is doing.
4. **Assemble**: the chaptered M4B is built.

A stalled job is restarted by itself and finished clips are kept. Tick the checkbox beside books to **hold, resume, remove, move** or **start now** for several at once. Set an [ntfy](https://ntfy.sh) topic to get a message when a book finishes or fails; replying `current`, `queue`, `done`, `now <number>`, `stop` or `send <number>` to the topic controls the queue from your phone.

### Done
Finished books. Here you can:

* **Rate** a book out of five stars, for your own record.
* See the **folder size**, how many clips it holds, and how many are **unused** (leftovers from older versions of the text or voices).
* **Delete unused clips**: removes only clips the book no longer uses, never ones it needs or that a correction can go back to.
* **Clean up the folder**: either delete the whole folder, or **zip everything except the clips** (the zip can be restored later; the next build remakes the clips).

Nothing is ever deleted automatically. A folder that a queued or running job uses is never touched. The finished audiobook itself is never touched.

## 4. Music

Open **④ Extras → Music**. Only free recordings are offered from the search (public domain or CC0, from Wikimedia Commons). You can also add your own.

* **Opening**: pick a track, a start point, and optionally a **spoken title** read over it (a new line starts a new part, after the pause you set).
* **Bridges**: short music before a section. Either one shared track at a spacing you choose, or your **own track per section**.
* **Ending**: music at the close.
* **▶ Preview** plays a short mp3 of the opening, a bridge or the ending, laid into clips already made. The final book will sound slightly different because clips may be remade, but it shows you the fit.
* **Add a recording from a link**: paste a direct https:// link to an audio file, name it, say where it came from. A file over 120 MB is re-encoded to Opus in the background to just fit.
* A credit line for every track used is added to the book.

Music files stay on your machine and are not part of the project's Git repository.

## 5. Voice lab (cloning a voice)

* Upload or record 5–12 seconds of clean speech. **Any audio file works** (wav, mp3, flac, ogg, opus, m4a, aac, even a video's soundtrack).
* Type what is said exactly, or let the app transcribe it.
* **Generate takes** makes several versions with different seeds side by side. **Save voice** keeps the clip, the words, the settings and the seed of the one you pick.
* **Take a reference from an audiobook**: open an `.m4b` or any audio file, choose a chapter, and the app cuts clean clips at pauses and writes down what is said (matching it to the ebook if you supply one).
* A saved voice can be used by any engine from the voice menus.
* **Make a Kokoro voice from a recording** searches for a fast Kokoro voice that sounds like your target.

Cloning quality depends on the clip's cleanliness and the engine, not on a higher bitrate.

## 6. Ask Claude

The **✨ Ask Claude** tab is a helper you talk to in plain language ("give the narrator a calmer voice", "put a march between the second and third parts", "make a cover"). It uses your own Claude Code login. It works on a **copy**; you review the changes as a difference and nothing happens until you press **Apply**.

It can edit voices and pacing, the pronunciation list, and the music plan. It can look at the book (status, chapters, names by where they were placed, unused clips), clean the pronunciation list, search for and add free music, make previews, and *ask* for a rebuild, a clean-up of unused clips, a name-origin check, or to hold or resume the book's job. Those requests also wait for **Apply**.

## 7. The finished audiobook

The file is in `out/` as a chaptered `.m4b`. By default the audio is **Opus at 64 kb/s**, normalised to **−16 LUFS** (comfortably loud on a phone). Opus plays in Android players, VLC and Audiobookshelf. If you use Apple Books, set `audio_codec: aac` in `config.yaml`. If you ticked *send to my phone*, the queue shares the file through KDE Connect once the book is done, retrying while the phone is away.

## 8. Sharing the GPU

Anything in the app that loads a model (remaking a clip, hearing a voice, finding genders) needs the graphics card. If a book is being made, the app asks you to press **Pause the book and use the GPU**. The book's speech stops (finished clips are kept) and carries on by itself a few minutes after you finish, or when you press **Give the GPU back to the queue**.

## 9. Troubleshooting

| Problem | What to try |
|---|---|
| A step is 🔒 | Do the numbered step before it. |
| Parser picked the wrong layout | **Book** tab: the note under the file says what it assumed; pick another reader. |
| A name sounds wrong | **③ Names**: edit the IPA or respelling, then **⑥ Fix → Rebuild**. |
| A name is in the wrong accent | **③ Names**: use the setting filter, correct the group, **re-check**. |
| One clip is bad | **⑥ Fix**: find it, change the words or seed, make a new take. |
| Job says it stalled | The queue restarts it by itself; finished clips are kept. |
| Disk is filling up | **Done** tab: delete unused clips, or zip a book without its clips. |
| A link will not add | It must be a direct https:// link to the audio file, not a web page. |
