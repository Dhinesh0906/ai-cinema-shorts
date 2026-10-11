"""Compare an independent Whisper transcript with the approved line: WER, missing/extra words."""
from __future__ import annotations

import re

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> list[str]:
    """Integer -> spoken words (Whisper writes digits; scripts usually spell numbers)."""
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + (num_words(n % 10) if n % 10 else [])
    if n < 1000:
        return num_words(n // 100) + ["hundred"] + (num_words(n % 100) if n % 100 else [])
    if n < 10000 and 1100 <= n < 2000 or 2010 <= n < 2100:  # years: 1987 -> nineteen eighty seven
        return num_words(n // 100) + (num_words(n % 100) if n % 100 else ["hundred"])
    if n < 1_000_000:
        return num_words(n // 1000) + ["thousand"] + (num_words(n % 1000) if n % 1000 else [])
    return [str(n)]

STOP = {"a", "an", "the", "and", "of", "to", "in", "on", "at", "is", "it", "its", "was", "for", "with", "as", "by"}


def norm_words(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("—", " ").replace("-", " ").replace(",", "")
    toks = re.findall(r"[a-z0-9']+", text)
    out = []
    for t in toks:
        t = t.strip("'")
        if t.endswith("'s"):
            t = t[:-2]
        if t.isdigit():
            out.extend(num_words(int(t)))
        elif t:
            out.append(t)
    return out


def align(ref: list[str], hyp: list[str]) -> tuple[int, list[str], list[str], list[tuple[str, str]]]:
    """Levenshtein alignment -> (edits, deleted ref words, inserted hyp words, substitutions)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, dels, ins, subs = n, m, [], [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                subs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels.append(ref[i - 1])
            i -= 1
        else:
            ins.append(hyp[j - 1])
            j -= 1
    return d[n][m], dels[::-1], ins[::-1], subs[::-1]


def sound_key(word: str) -> str:
    """Crude phonetic key for invented names: Kael = Kale = Cael = Kayel -> 'kl'."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    for a, b in (("ph", "f"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "s"), ("x", "ks")):
        w = w.replace(a, b)
    w = w[0] + re.sub(r"[aeiouyhw]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", w)


def compare(expected: str, transcript: str, names: list[str] | tuple[str, ...] = ()) -> dict:
    """names: invented proper nouns (characters, places). Whisper can't know their spelling, so a heard word
    that sounds like one (same phonetic key) counts as that name."""
    ref, hyp = norm_words(expected), norm_words(transcript)
    keys = {sound_key(n): n for name in names for n in norm_words(name) if len(n) >= 3}
    hyp = [w if w in ref or sound_key(w) not in keys or keys[sound_key(w)] not in ref else keys[sound_key(w)]
           for w in hyp]
    edits, dels, ins, subs = align(ref, hyp)
    wer = edits / max(1, len(ref))
    missing_content = [w for w in dels if w not in STOP] + [r for r, _ in subs if r not in STOP]
    return {"wer": round(wer, 3), "ref_words": len(ref), "hyp_words": len(hyp), "missing": dels, "extra": ins,
            "substitutions": subs, "missing_content_words": missing_content}
