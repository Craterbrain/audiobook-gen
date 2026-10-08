"""IPA -> plain-English respelling for Chatterbox, which reads ordinary spelling far better than bare IPA.
The spelling chosen for each sound lives in OPTIONS; `tune()` in work/ picks them by speaking names through
Chatterbox and scoring what it said against the IPA (see respell_check.py)."""
import re

TOKENS = ["tʃ", "dʒ", "eɪ", "aɪ", "oʊ", "ɔɪ", "aʊ"]
SHORT = set("æɛɪʌ")          # short stressed vowels: double the following consonant to keep them short
VOWELS = set("aeiouæɛɪʊʌɑɔəɚɝ")

# the choices below are the tunable part: option name -> the spelling used
OPTIONS = {
    "schwa_init": "a", "schwa_mid": "uh", "schwa_fin": "uh",
    "ai": "eye", "ai_fin": "y", "ei": "ay", "i": "ee", "ou": "oh", "ah": "ah", "ae": "a", "eh": "e", "ih": "i",
    "uh": "u", "oo": "oo", "aw": "aw", "er": "er", "oi": "oy", "au": "ow",
    "double": True, "hiatus": "", "ar": "ar", "or": "or", "air": "air", "eer": "eer",
}
CHOICES = {
    "schwa_init": ["a", "uh", "eh", "ah"], "schwa_mid": ["uh", "a", "eh", "i"], "schwa_fin": ["uh", "a", "ah"],
    "ai": ["eye", "y", "igh", "ie"], "ai_fin": ["y", "eye", "igh", "ie"], "ei": ["ay", "ai", "a"], "i": ["ee", "e", "ea"],
    "ou": ["oh", "o", "oe"], "ah": ["ah", "o", "a"], "ae": ["a", "ah", "aa"], "eh": ["e", "eh"], "ih": ["i", "ih", "e"],
    "uh": ["u", "uh", "o"], "oo": ["oo", "u"], "aw": ["aw", "o", "au"], "er": ["er", "ur", "ir"],
    "double": [True, False], "hiatus": ["", "-", "y"], "ar": ["ar", "ahr"], "or": ["or", "ore"], "air": ["air", "er", "ar"],
    "eer": ["eer", "ear", "ir"],
}
CONS = {"ɹ": "r", "r": "r", "ɡ": "g", "g": "g", "ʃ": "sh", "ʒ": "zh", "θ": "th", "ð": "th", "tʃ": "ch", "dʒ": "j", "ŋ": "ng",
        "j": "y", "k": "k", "x": "ch", "ɾ": "t", "ɫ": "l", "ʔ": "", "h": "h"}


def tokenize(ipa: str) -> list[tuple[str, bool]]:
    """[(phoneme, stressed)] with stress marks, length marks and tie bars removed."""
    ipa = re.sub(r"[/.\s͡‿‧·]", "", ipa).replace("ː", "")
    out, i, stress = [], 0, False
    while i < len(ipa):
        c = ipa[i]
        if c == "ˈ":
            stress = True; i += 1; continue
        if c == "ˌ":
            i += 1; continue
        t = next((x for x in TOKENS if ipa.startswith(x, i)), c)
        out.append((t, stress and (t in VOWELS or t in ("eɪ", "aɪ", "oʊ", "ɔɪ", "aʊ")))); i += len(t)
        if out[-1][1]:
            stress = False
    return out


def respell(ipa: str, name: str = "", opt: dict | None = None) -> str:
    o = {**OPTIONS, **(opt or {})}
    ph = tokenize(ipa)
    out: list[str] = []
    n = len(ph)
    isv = lambda t: t[0] in VOWELS or t[0] in ("eɪ", "aɪ", "oʊ", "ɔɪ", "aʊ", "i", "u", "e", "o", "a")
    k = 0
    while k < n:
        p, st = ph[k]
        nxt = ph[k + 1][0] if k + 1 < n else ""
        first, last = k == 0, k == n - 1
        s = None
        if p in ("ɑ", "ɔ", "ɛ", "ɪ", "ʊ") and nxt in ("ɹ", "r"):
            s = {"ɑ": o["ar"], "ɔ": o["or"], "ɛ": o["air"], "ɪ": o["eer"], "ʊ": "oor"}[p]; k += 1
        elif p in ("ɝ", "ɚ"):
            s = o["er"]
        elif p == "ə":
            s = o["schwa_init"] if first else o["schwa_fin"] if last else o["schwa_mid"]
        elif p == "aɪ":
            s = o["ai_fin"] if last else o["ai"]
        elif p == "eɪ": s = o["ei"]
        elif p in ("i", "iː"): s = o["i"]
        elif p == "oʊ": s = o["ou"]
        elif p == "ɑ": s = o["ah"]
        elif p == "æ": s = o["ae"]
        elif p == "ɛ": s = o["eh"]
        elif p == "ɪ": s = o["ih"]
        elif p == "ʌ": s = o["uh"]
        elif p in ("u", "ʊ"): s = o["oo"]
        elif p == "ɔ": s = o["aw"]
        elif p == "ɔɪ": s = o["oi"]
        elif p == "aʊ": s = o["au"]
        elif p in ("e",): s = o["ei"]
        elif p in ("o",): s = o["ou"]
        elif p in ("a",): s = o["ah"]
        else:
            s = CONS.get(p, p)
        vowel = p in VOWELS or p in ("eɪ", "aɪ", "oʊ", "ɔɪ", "aʊ", "i", "u")
        if out and vowel and o["hiatus"] and prev_vowel:
            s = o["hiatus"] + s
        # a stressed short vowel followed by one consonant and another vowel: double the consonant (Hadad -> Haddad)
        out.append(s)
        if o["double"] and p in SHORT and st and k + 2 < n and ph[k + 1][0] not in VOWELS and isv(ph[k + 2]) and ph[k + 1][0] not in ("ɹ", "h", "j", "ʃ", "ʒ", "tʃ", "dʒ", "θ"):
            out.append(CONS.get(ph[k + 1][0], ph[k + 1][0]));
        prev_vowel = vowel
        k += 1
    return "".join(out).capitalize()
