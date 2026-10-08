# Third-party data and models

The project's own code is MIT-licensed (see `LICENSE`). The notes below cover data and models that are not covered by that license.

This project ships code and a few small data files. Everything else is downloaded on demand or supplied by you.

## Data included in the repository
- `data/bible_ipa.json` — IPA pronunciations for Bible names and places. It combines entries from **Wiktionary** and
  **WikiPron** (CC BY-SA 4.0 / CC BY-SA), **ipa-dict** (MIT), words written by hand for this project, and the user-typed
  corrections in `data/bible_ipa_user.txt`. Because it contains Wiktionary-derived data, treat the file as
  **CC BY-SA 4.0** and keep this attribution when you reuse it. Each entry names its source.
- `data/bible_ipa_claude_*.txt` — the hand-written readings (unreviewed; check by ear).
- `samples/` — public-domain texts (KJV John, Macbeth, excerpts of *The Count of Monte Cristo*).

## Not included (bring your own, or it downloads on first use)
- Bible texts under copyright (for example the NASB). `python -m audiobook_gen bible-dict <verse-per-line text>` builds the
  dictionary from any text you own; the NASB itself is not distributed here.
- Your audiobooks, ebooks, recordings and cloned voices (`voices/`, `work/`, `out/`).
- Model weights: Kokoro-82M, F5-TTS, Chatterbox, Qwen3-TTS, Whisper, WavLM and others are downloaded from Hugging Face
  and keep their own licenses. Check each before commercial use (for example, F5-TTS weights are released for
  non-commercial use).
- Downloaded reference data: WikiPron tables, ipa-dict word lists, S. V. R. Ford's *Pronouncing Vocabulary of Scripture
  Proper Names* (1900, public domain) — fetched into `data/` when needed.
