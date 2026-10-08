"""Small-LLM tie-breaker for quotes the structural decoder is unsure about.

Not generation: one forward pass per quote, comparing the model's probability for the letter of
each candidate speaker (multiple choice). The log-probs are fed back into the Viterbi decoder as
extra evidence, so structure (turn-taking, genders, explicit tags) and language both vote.
Default model: Qwen2.5-1.5B-Instruct in bf16 (~3 GB VRAM); freed again after parsing.
"""
import math

LETTERS = "ABCDEFGHIJKL"


class Judge:
    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", device_pref: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .tts.device import pick_device

        self.device = pick_device(device_pref)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16 if self.device != "cpu" else torch.float32).to(self.device).eval()
        self.ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
        self.calls = 0

    def choose(self, system: str, user: str, options: list[str], debias: bool = False) -> dict[str, float]:
        """Multiple choice by letter probability: log P(letter) per option, renormalised over the options.
        debias=True also asks with the options reversed and averages, cancelling the model's lean
        towards particular letters."""
        options = options[:len(LETTERS)]
        if debias:
            fwd, rev = self.choose(system, user, options), self.choose(system, user, options[::-1])
            avg = {o: (fwd[o] + rev[o]) / 2 for o in options}
            z = math.log(sum(math.exp(v) for v in avg.values()))
            return {o: v - z for o, v in avg.items()}
        import torch

        options = options[:len(LETTERS)]
        listing = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"{user}\n{listing}\nAnswer with a single letter."}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits[[self.ids[i] for i in range(len(options))]], dim=0).cpu().tolist()
        self.calls += 1
        return dict(zip(options, lp))

    def score(self, context: str, quote: str, options: list[str]) -> dict[str, float]:
        """Who speaks `quote`? log-probs over candidate speakers."""
        return self.choose(
            "You are an expert reader of novels. You work out which character speaks a given line of "
            "dialogue, using who is present, who was just addressed, pronouns and gender, and the natural "
            "back-and-forth of conversation.",
            f"Passage:\n{context}\n\nThe line to attribute: \u201c{quote}\u201d\n\nWho speaks that line?",
            options)

    SYS_LANG = ("You are a linguist. You identify the language of origin of words and names so a "
                "text-to-speech engine can pronounce them correctly.")

    def is_foreign(self, word: str, sentence: str) -> float:
        """P(word keeps a foreign-language pronunciation) rather than being read as ordinary English."""
        opts = ["An English word, or the ordinary English form of a name",
                "A foreign word or name that keeps its original-language spelling and pronunciation"]
        lp = self.choose(self.SYS_LANG, f'Sentence: {sentence}\n\nIs "{word}" in this sentence:', opts, True)
        return math.exp(lp[opts[1]])

    GROUPS = {"Narrator", "Crowd", "Guest", "Guests", "All", "Servants", "Officers", "Jews", "Disciples", "Pharisees",
              "Murderers", "Both Murderers", "Lords", "Soldiers", "Attendants", "Voices"}

    def genders(self, book: str, names: list[str], progress=None) -> dict[str, tuple[str, float]]:
        """{name: (male|female|unknown, probability)} for the book's characters. The question follows the book's
        own context ("In the book X these A, B, C are all the characters found..."), then asks about each name by
        letter probability (debiased), which is steadier on a 1.5B model than asking it to write a list."""
        people = [n for n in names if n not in self.GROUPS]
        intro = (f'In the book "{book}" these {", ".join(people)} are all the characters found. '
                 "I would like to know the gender of these characters.")
        opts = ["Male", "Female", "Not a single person, or unknown"]
        out = {n: ("unknown", 1.0) for n in names if n in self.GROUPS}
        for k, n in enumerate(people):
            if progress:
                progress(k / max(1, len(people)), f"Asking about {n} ({k + 1}/{len(people)})")
            lp = self.choose("You are an expert on literature and know the characters of well-known books.",
                             f"{intro}\n\nWhat is the gender of {n}?", opts, True)
            best = max(lp, key=lp.get)
            out[n] = (("male", "female", "unknown")[opts.index(best)], math.exp(lp[best]))
        return out

    BOOK_LANGS = [("en", "English"), ("fr", "French"), ("it", "Italian"), ("es", "Spanish"), ("de", "German"),
                  ("pt", "Portuguese"), ("nl", "Dutch"), ("la", "Latin"), ("ru", "Russian"), ("el", "Greek")]

    def book_language(self, title: str, author: str) -> tuple[str, float]:
        """Language of the characters' and places' names, judged from title and author alone. Reliable
        where per-word guessing is not (Monte Cristo -> French, Macbeth -> English)."""
        lp = self.choose(
            "You are a linguist helping a text-to-speech engine pronounce names in a book. Assume the "
            "names are English unless the book is clearly set among speakers of another language.",
            f'Book: "{title}"' + (f" by {author}" if author else "") +
            ".\n\nWhat language are the characters' names and the place names in this book mainly from?",
            [n for _, n in self.BOOK_LANGS], True)
        best = max(lp, key=lp.get)
        return dict((n, c) for c, n in self.BOOK_LANGS)[best], math.exp(lp[best])

    def close(self):
        import gc

        import torch
        del self.model
        gc.collect()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()


def context_for(paras: list[str], pi: int, quote: str, before: int = 2, limit: int = 1400) -> str:
    """Previous paragraphs + the paragraph holding the quote, trimmed to the end."""
    text = "\n\n".join(paras[max(0, pi - before):pi + 1])
    return text[-limit:]
