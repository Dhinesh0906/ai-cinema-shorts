"""Shorts-style animated captions rendered as ONE transparent video track.

Style: 2-3 words at a time, heavy uppercase font, white fill, thick black outline + soft shadow, the
word currently being spoken highlighted (yellow) using Whisper word timestamps, and a short "pop"
(slightly larger for the first frames) when a new group appears. Centered at `y_center` of the frame
height (default 0.72 — between the middle and the bottom, clear of the Shorts UI overlay).

FFmpeg here has no libass/drawtext, so frames are drawn with Pillow and stitched with the concat
demuxer into a PNG-in-MOV track with alpha; the renderer overlays it once.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..schemas import Caption

FONT_FILE = "/System/Library/Fonts/Avenir Next.ttc"
FONT_INDEX_HEAVY = 8
FALLBACK_FONT = "/System/Library/Fonts/Supplemental/Arial Black.ttf"


@dataclass(frozen=True)
class CaptionStyle:
    size: int = 96
    min_size: int = 72
    y_center: float = 0.72
    fill: tuple = (255, 255, 255, 255)
    highlight: tuple = (255, 221, 0, 255)
    stroke: tuple = (0, 0, 0, 255)
    stroke_width: int = 9
    shadow_offset: int = 6
    shadow_blur: int = 10
    max_width_frac: float = 0.86
    pop_scale: float = 1.12
    pop_seconds: float = 0.09
    uppercase: bool = True


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_FILE, size, index=FONT_INDEX_HEAVY)
    except OSError:
        return ImageFont.truetype(FALLBACK_FONT, size)


def _layout(words: list[str], font, draw, max_w: float, space: float) -> list[list[int]]:
    lines, cur, w = [], [], 0.0
    for i, word in enumerate(words):
        ww = draw.textlength(word, font=font)
        if cur and w + space + ww > max_w:
            lines.append(cur)
            cur, w = [], 0.0
        w = ww if not cur else w + space + ww
        cur.append(i)
    if cur:
        lines.append(cur)
    return lines


def draw_caption(words: list[str], active: int | None, W: int, H: int, style: CaptionStyle,
                 scale: float = 1.0) -> Image.Image:
    words = [w.upper() if style.uppercase else w for w in words]
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    size = int(style.size * scale)
    while True:  # prefer one line: shrink down to min_size before allowing a wrap
        font = _font(size)
        space = d.textlength(" ", font=font)
        lines = _layout(words, font, d, W * style.max_width_frac, space)
        if len(lines) == 1 or size <= int(style.min_size * scale):
            break
        size -= 4
    asc, desc = font.getmetrics()
    line_h = int((asc + desc) * 1.05)
    top = int(H * style.y_center - line_h * len(lines) / 2)
    placed = []
    for li, idxs in enumerate(lines):
        widths = [d.textlength(words[i], font=font) for i in idxs]
        x = (W - (sum(widths) + space * (len(idxs) - 1))) / 2
        y = top + li * line_h
        for i, wd in zip(idxs, widths):
            placed.append((i, x, y))
            x += wd + space
    # soft drop shadow
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for i, x, y in placed:
        sd.text((x + style.shadow_offset, y + style.shadow_offset), words[i], font=font, fill=(0, 0, 0, 170),
                stroke_width=style.stroke_width, stroke_fill=(0, 0, 0, 170))
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(style.shadow_blur)))
    d = ImageDraw.Draw(canvas)
    for i, x, y in placed:
        d.text((x, y), words[i], font=font, fill=style.highlight if i == active else style.fill,
               stroke_width=style.stroke_width, stroke_fill=style.stroke)
    return canvas


def _segments(captions: list[Caption], duration: float, style: CaptionStyle):
    """(start, end, words, active_index, scale) segments covering the timeline; None words = blank."""
    segs, t = [], 0.0
    for c in sorted(captions, key=lambda c: c.start):
        if c.start > t + 1e-3:
            segs.append((t, c.start, None, None, 1.0))
        words = [w.word for w in c.words] or c.text.split()
        times = [(w.start, w.end) for w in c.words]
        start = max(c.start, t)
        if not times:
            segs.append((start, c.end, words, None, 1.0))
        else:
            for k in range(len(words)):
                a = start if k == 0 else times[k][0]
                b = times[k + 1][0] if k + 1 < len(words) else c.end
                if b - a <= 1e-3:
                    continue
                if k == 0 and b - a > style.pop_seconds:
                    segs.append((a, a + style.pop_seconds, words, 0, style.pop_scale))
                    a += style.pop_seconds
                segs.append((a, b, words, k, 1.0))
        t = max(t, c.end)
    if t < duration:
        segs.append((t, duration, None, None, 1.0))
    return segs


def build_track(captions: list[Caption], duration: float, out: Path, W: int = 1080, H: int = 1920,
                fps: int = 24, style: CaptionStyle = CaptionStyle()) -> Path:
    work = out.parent / "caption_frames"
    work.mkdir(parents=True, exist_ok=True)
    blank = work / "blank.png"
    Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(blank)
    cache: dict = {}
    lines = []
    for a, b, words, active, scale in _segments(captions, duration, style):
        if words is None:
            path = blank
        else:
            key = (tuple(words), active, scale)
            if key not in cache:
                p = work / f"cap_{len(cache):04d}.png"
                draw_caption(words, active, W, H, style, scale).save(p)
                cache[key] = p
            path = cache[key]
        lines.append(f"file '{path}'\nduration {max(b - a, 1 / fps):.4f}\n")
    lines.append(f"file '{blank}'\n")  # concat demuxer needs the last file repeated
    listing = work / "captions.txt"
    listing.write_text("".join(lines))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-vf", f"fps={fps},format=rgba", "-t", f"{duration:.3f}", "-c:v", "png", str(out)],
                   check=True)
    return out
