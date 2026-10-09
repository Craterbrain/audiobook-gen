# audiobook-gen

Local pipeline: `.epub`/`.txt` -> speaker-parsed segments -> pronunciation lexicon -> Kokoro-82M / F5-TTS on an Intel Arc GPU (PyTorch XPU) -> chapterized, tagged `.m4b`.

## Setup
```
./setup.sh      # .venv (Python 3.12 via uv bootstrap if absent), torch XPU wheels, deps, en_core_web_sm
```
Manual equivalent:
```
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt && pip uninstall -y torchcodec
python -m spacy download en_core_web_sm
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```
System needs: `ffmpeg`, Intel GPU compute runtime (level-zero / intel-compute-runtime).

## Run
```
./run_sample.sh                                   # John 1-3 (KJV) -> out/john_kjv.m4b
python -m audiobook_gen run book.epub             # everything
python -m audiobook_gen {extract|parse|lexicon|synth|assemble} book.epub
```
Stages write to `work/<name>/` (`chapters.json`, `segments.json`, `lexicon.json`, `clips/`, `chapters/`) and `synth` caches clips, so reruns are cheap.

## Customising
- **Voices**: `config.yaml`. Kokoro roles take a preset `voice`; F5 roles take `ref_audio` (5-15 s clip) + `ref_text` for zero-shot cloning. Unlisted characters get a stable voice from a pool.
- **Pronunciation**: edit `work/<name>/lexicon.json` (`ipa` for Kokoro, `respell` for F5); re-running `lexicon` preserves edits. Entries are sorted by frequency.
- **Speaker labels**: heuristic by default (handles quoted dialogue and KJV-style unquoted speech); `--llm-endpoint http://localhost:8080/v1/chat/completions` refines with a local LLM.
- **OpenVINO**: not wired in; the XPU PyTorch path is the supported one.

## Notes
- Verified on Arc B580: Kokoro RTF ~0.22 (John 1-3, ~14 min audio), F5 cloning works on XPU.
- `torchcodec` is removed because its PyPI wheel needs CUDA libs; the F5 engine patches `torchaudio.load` to use soundfile.

## GUI

**One-click launch.** `launch.sh` starts the GUI if it isn't running, waits until it answers, and opens it in Firefox. It is installed as an *Audiobook Gen* app-menu entry (with a *Stop the server* action) and a desktop icon (`~/.local/share/applications/audiobook-gen.desktop`, `~/Desktop/Audiobook Gen.desktop`, icon in `assets/`). `stop.sh` stops the GUI server only; generation jobs are separate processes.
```
./gui.sh      # opens http://127.0.0.1:7860
```
Tabs: **Book** (upload, chapter tick-list, title/author/cover) → **Cast** (parse speakers, per-role engine/voice/speed table, F5 clone upload, voice preview, editable segments) → **Lexicon** (build + edit IPA/respellings) → **Generate** (pacing sliders, an *engines* panel with emotion-from-text for Chatterbox, how names are spoken — real IPA / respelling / plain — and the quote/tag pauses; live progress, Stop, M4B download). Built on Gradio, which F5-TTS already installs, so there are no new dependencies.

### Speaker attribution (quoted prose)
`attribution.py` decodes every quote in a chapter jointly (Viterbi) from: explicit "said X" phrases, pronoun gender, names being addressed inside a quote, turn-taking / A-B-A alternation, split quotes, and a discovered cast (places and bystanders can't be chosen). Hand-labelled check: `python tests/eval_attribution.py` (currently 88% over 129 quotes; ~84% on the held-out chapter).
In the GUI's Cast tab you can filter by character, tick lines, hear them in any voice, move them to another (or new) character, or merge a whole character into another. Aliases / phrase roles (e.g. "the old man" → Old Dantès) are editable there too.

### Tie-breaker model
`python -m audiobook_gen parse book.txt --judge` (GUI: Cast tab checkbox) loads Qwen2.5-1.5B-Instruct in bf16 (~3 GB VRAM) and asks it a multiple-choice question *only* for quotes the structural decoder is unsure about (posterior < 0.8). Its letter probabilities are added to the Viterbi scores at weight 0.5, then the model is unloaded. Qwen2.5-3B was tested and was no better here; 4-bit loading isn't available on XPU without a Level-Zero toolchain.

### Lexicon
A word is flagged when Kokoro's own G2P (misaki) has no dictionary entry for it, so names, foreign words and archaic terms are all found without frequency noise. Each entry shows **guess** (what Kokoro says now) and a language hint; `--lang fr` (or `name_lang: fr` in the config) auto-fills IPA from espeak in that language, restricted to symbols Kokoro knows. Abbreviations (M., Mme., Dr., St.) and roman numerals (Louis XVIII, Act II) are expanded before synthesis, and possessives (Edmond's) keep their suffix.

#### Language detection (English first)
1. **Book language** comes from the small LLM reading the **title and author** (Monte Cristo → French, Macbeth → English; ~1.0 confidence both). A user-set language wins; an English verdict locks the book to English. Only fr/it/es/de/pt/nl are accepted (Latin-script names espeak can voice).
2. **Per word**, `wordfreq` evidence (a language knows the word better than English does) is used first. Names, places and accented words that match nothing take the book's language; plain lowercase words must also pass the LLM's debiased "ordinary English, or foreign?" check. With no book language, everything stays English.
3. **Place-name phrases** (Allées de Meilhan, Fort Saint Nicholas, Palais de Justice, La Réserve) are stored and pronounced **whole**, so liaison and elision come out right. Both an IPA and an English-style respelling (`ah-LAY duh meh-ee-LAHN`) are returned. Phrases built from English words (Island of Elba) are left to the individual words.
The model is never asked to name the language of a lone word: it guessed Latin for Macbeth's Scottish names, and told French that *overtook* was French once it was primed with a French book. IPA and respelling come from espeak plus a rule-based converter, not from the LLM.

### Clone tab

**Three cloning engines.** The Clone tab's *Make the takes with* switch picks **F5-TTS**, **Chatterbox** or **Qwen3-TTS**; Chatterbox shows its own emotion/pace sliders (F5's sliders hide), and a saved voice remembers them. Any saved clone can be voiced by any engine: the voice menus list it as *(F5)*, *(Chatterbox)* and *(Qwen3)*, *Assign to role* takes an engine, and *Hear a saved voice in every engine* plays the same sentence in all three. Install them with `./setup_engines.sh chatterbox|qwen3|all` (each gets its own virtualenv and reuses the main PyTorch XPU build). Chatterbox and Qwen3 run as worker processes in their own environments (`.venv-chatterbox`, `.venv-tts2`); the engines panel on the Generate tab says whether each is installed. *Generate → Advanced → Chatterbox workers* runs two Chatterbox processes at once (measured about 22% faster, because one worker already keeps the GPU about 78% busy); not recommended for GPUs with less than 16 GB of VRAM, since each worker can hold 4–8 GB. The Lexicon tab's lookup has *Bible book* (the Bible IPA dictionary) and *Offline only* options.
Clone a voice with F5-TTS: upload or record a 5–12 s clean clip, type (or auto-transcribe) exactly what is said, and tune **pace** (the voice's default cadence), quality steps, guidance, sway sampling, cross-fade, loudness and **seed**. "Generate takes" renders up to three seeds side by side; **Save voice** keeps the clip, transcript, settings and the seed of the take you pick in `voices/library/<name>/`. Roles use saved voices by name (`{engine: f5, library: <name>}`), via "Assign to role" or Cast → Engine `f5` + Voice `<name>`. A fixed seed keeps a character's timbre steady across a whole book.

#### References from an audiobook
Clone tab → "Take a reference from an audiobook": open a `.m4b`, read its chapters, pick one, optionally add the matching ebook. Clean 6–12 s clips are cut at pauses found in the waveform (no timestamps needed); Whisper writes down each short clip as plain text; with an ebook, that text is fuzzy-matched (3-gram anchors + difflib) to find the book's exact words, which become the reference transcript. The status line says which ebook chapter the audio came from and how closely the clip matched. "Use as reference" fills the cloner's clip and transcript fields. Without an ebook you get Whisper's text, to check by hand.

### Choosing voices (Cast tab)
Pick **Narration style**. *Different voice for each character* shows one row per character with a voice menu (Kokoro presets with readable names, plus your saved clones), a pace slider and a **▶ Hear** button; changes save as you make them. *One narrator reads everything* shows a single narrator menu: every character and the narration are read by that voice as continuous text (stored as `single_voice` in the project's `config.yaml`). Saved a new clone? Press "Refresh voice list".

#### Auto-spelling for each engine
Kokoro takes IPA; F5 reads plain text, so it needs a respelling. Both are now filled automatically: IPA from espeak in the word's language (or Kokoro's own guess for English), and a respelling made from that IPA (`dɑ̃ɡlˈaʁ` → `dahn-GLAHR`, `kˈAdɹWs` → `KAY-drows`; stressed syllable in capitals). If you type or edit an IPA and leave the respelling blank, F5 still gets one made on the fly, and saving the lexicon refreshes respellings that were made automatically. A respelling you typed yourself is never overwritten. Well-known words (Macbeth, Fife) are skipped unless you give them an IPA. The Lexicon tab has an **Auto-spell for F5** button that fills or refreshes them on demand.

### Speed
- **F5 half precision** (`f5_precision: float16`, default; GUI checkbox on Generate): F5 loads in float32 on Intel GPUs, which was 1.40× slower than real time; float16 runs at 0.29–0.33× with the same words, length and loudness (bfloat16 fails in the ODE solver). If half precision ever returns invalid audio the engine switches itself back to float32.
- **Parallel workers** (`workers: {kokoro: 4, f5: 1}`, GUI slider): each worker thread has its own model copy. Kokoro on John 1: 1 worker 59 s, 3 workers 23 s, 4 workers 17 s. F5 gained nothing from extra workers (the GPU is already saturated), so it defaults to 1.
- Checking the GPU is used: `intel_gpu_top` doesn't support the Xe driver, but `cat /proc/<pid>/fdinfo/*` shows `drm-resident-vram0` (~3.7 GB while F5 runs). One CPU core at ~100% is the thread feeding the GPU, not the computation.

### Finding a spelling for F5 (Lexicon tab → "Search for the best F5 spelling")
Give a word, its IPA, and a clone voice. It tries 20–100 generated spellings, has a phoneme recognizer write down what F5 said, and ranks them by closeness to the IPA; click a row to hear it and send it to the respelling box. Stress isn't scored, so listen to the top few.

### Genders (Cast tab)
**Find genders with the small model (Qwen)** asks, with the book's title: *In the book "<title>" these <names> are all the characters found. I would like to know the gender of these characters.* and then, per character, "What is the gender of <name>?" (answered by letter probability, so a 1.5B model stays reliable). Groups (Crowd, Guests, Jews…) are marked unknown without asking. Each character row has a male/female/unknown menu you can correct; the result is stored as `genders` in the project's `config.yaml`. It is used to (1) pick default voices for characters you haven't chosen (`Assign voices by gender` gives each a distinct voice of their gender and keeps clones), and (2) tell the attribution step who is "he" or "she" the next time you parse.

### Kokoro voices from a recording (Clone tab → "Make a Kokoro voice from a recording")
Kokoro has no audio encoder, so a voice pack is *searched*: start from the stock voice that scores closest to the target (or from a voice pack you made earlier, if it scores better), then nudge one shared 256-number offset with an evolution strategy, keeping changes that raise the WavLM speaker-similarity (`microsoft/wavlm-base-plus-sv`) of Kokoro's speech to the target. Targets: a saved clone voice (its reference clip plus a few F5 sentences), an audio file, or a random snippet near any point of an `.m4b` (clean 7–12 s clips cut at pauses). Output: `voices/kokoro/<name>.pt` (`[510,1,256]`, an ordinary Kokoro voice) plus a `.json` report with similarity before/after, including on sentences it was never tuned on. Packs appear in the voice menus as `kokoro:pack:<name>`, take IPA, and run far faster than F5. Command-line: `audiobook_gen.voicepack.search(...)`. `voicepack.refine(name, target_clips)` then measures the finished voice on unseen sentences and stores corrections in its json: a pitch trim (cents), a formant shift (vowel colour, pitch untouched), a calibrated speed, and silence at clip edges trimmed. Pitch is measured on loud frames only; room noise and silence otherwise drag the estimate down.


**Looking up real IPA.** `python -m audiobook_gen lookup x --work work/<book> --config <cfg> [--lang fr] [--bible] [--offline] [--wikis lotr]` (or *Look up IPA online* in the Lexicon tab) fills empty IPA from, in order: the Bible dictionary and Ford's 1900 scripture-names guide (with `--bible`), Wiktionary's `{{IPA}}` templates, ipa-dict (MIT), WikiPron, then any Fandom wikis you name. `--offline` skips the network. Entries you typed are never touched; online answers and misses are cached in `data/pronunciation_cache.json`. Chatterbox and Qwen3 read the IPA directly (`text_lexicon: rawipa`).

**Bible IPA dictionary.** `python -m audiobook_gen bible-dict nasb1995.txt` builds `data/bible_ipa.json`: every capitalised name and place in a verse-per-line Bible text (3,152 for the NASB), each with `ipa`, `source`, `count` and first reference. Hand-written readings live in `data/bible_ipa_claude_*.txt` (Claude's, unreviewed) and `data/bible_ipa_user.txt` (yours, wins over everything); both are plain `Name|IPA` lines, and `bible-dict apply` re-applies them in a moment.

## Licenses and third-party data
The code is released under the [MIT License](LICENSE). The bundled Bible IPA dictionary contains Wiktionary-derived data and is CC BY-SA 4.0; see [NOTICE.md](NOTICE.md) for that and for model licenses. Your books, recordings and cloned voices stay on your machine (they are git-ignored); the NASB text is not included.

## Queue, schedule and assistant
- **Generate tab** adds the book to the queue; set a start time, or tick *Only run overnight* (for example 23:00–06:30), before pressing it.
  The **Queue tab** shows what is queued and running. One runner
  makes the queued books one at a time and watches each: a stalled or crashed job is restarted, it waits while anything
  else uses the GPU, and a supervisor restarts the runner itself. From the command line:
  `python -m audiobook_gen.jobqueue add --work work/<book> --not-before "2026-10-09 23:00" --window 23:00-06:30`, then `list` or `cancel <id>`.
- `./setup_queue_service.sh` installs the queue as a systemd user service so it starts at boot and is restarted if it ever
  dies (`--remove` uninstalls it). Without it, the app starts the queue whenever it opens.
- **Send to your phone** — tick *Send the finished audiobook to my phone* on the Generate tab (or use the buttons on the Queue tab) and the
  queue shares each finished book through KDE Connect, retrying every 5 minutes while the phone is out of reach.
- **Cover tab** — pick a picture (upload your own, or search public-domain / CC0 pictures on Wikimedia Commons), choose *picture window*
  (people, on a book-cloth colour) or *full background* (scenery), preview, and press *Use this cover*. `python -m audiobook_gen.covers` does the same from the command line.
- **Assistant tab** — ask Claude (through your Claude Code login) to set up voices, pacing or pronunciations, or to make a cover. It edits a copy,
  the only command it may run is the cover tool, you review what it proposes, and nothing changes until you press Apply.
