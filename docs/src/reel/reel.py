#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic). Doc tooling for the README reel.
"""reel.py CONFIG.json OUT_DIR -- silent reel for a studio README: frames as PNG, then ffmpeg (title card, a camera
move over each sheet or page spread with a caption, a wall of every sheet, the end card). Images: <work>/hi/<name>.png
(zoomed shots) and <work>/png/*.png (the wall), work = the folder of CONFIG.json."""
import json
import glob
import math
import os
import sys
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFilter, ImageFont

Image.MAX_IMAGE_PIXELS = None
CFG = json.load(open(sys.argv[1]))
HERE = os.path.dirname(os.path.abspath(sys.argv[1]))
OUT = sys.argv[2]
FW, FH, FPS = 1920, 1080, 30
NAVY0, NAVY1 = (14, 26, 52), (24, 42, 80)
GOLD, WHITE, MUTED = (214, 170, 82), (245, 246, 250), (170, 184, 210)
FD = os.environ.get("REEL_FONTS", "/usr/share/fonts/truetype/liberation2/")
F = lambda b, s: ImageFont.truetype(FD + ("LiberationSans-Bold.ttf" if b else "LiberationSans-Regular.ttf"), s)

FULL = tuple(CFG.get("full", (0.5, 0.5, 1.30)))   # (cx, cy as fractions of the sheet; view width as a fraction of sheet width)
SEGS = [(g["img"], g["title"], g["sub"], g["tag"], [((tuple(v) if v != "FULL" else FULL), h) for v, h in g["keys"]]) for g in CFG["segs"]]
MOVE, FADE = 1.25, 0.4
TITLE_S, WALL_S, END_S = 3.4, 4.0, 4.6


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def bg():
    g = Image.new("RGB", (1, FH))
    for y in range(FH):
        k = y / (FH - 1)
        g.putpixel((0, y), tuple(round(NAVY0[i] + (NAVY1[i] - NAVY0[i]) * k) for i in range(3)))
    return g.resize((FW, FH))


BG = None
_sheets = {}


def sheet(name):
    if name not in _sheets:
        if len(_sheets) >= 2:
            _sheets.pop(next(iter(_sheets)))
        _sheets[name] = Image.open(os.path.join(HERE, "hi", name + ".png")).convert("RGB")
    return _sheets[name]


def seg_len(keys):
    return sum(h for _, h in keys) + MOVE * (len(keys) - 1)


def view_at(keys, t):
    """camera (cx, cy, w) at time t within a segment: hold, move, hold, …"""
    acc = 0.0
    for i, (v, h) in enumerate(keys):
        if t < acc + h or i == len(keys) - 1:
            if i == len(keys) - 1:          # slow drift-in during the last hold
                k = min(1.0, (t - acc) / max(h, 1e-6))
                return (v[0], v[1], v[2] * (1 - 0.04 * k))
            return v
        acc += h
        if t < acc + MOVE:
            k = ease((t - acc) / MOVE)
            a, b = v, keys[i + 1][0]
            w = math.exp(math.log(a[2]) * (1 - k) + math.log(b[2]) * k)
            # keep the zoom feeling straight: move the centre in step with the scale
            s = (a[2] - w) / (a[2] - b[2]) if abs(a[2] - b[2]) > 1e-6 else k
            return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, w)
        acc += MOVE
    return keys[-1][0]


def camera(im, v):
    W, H = im.size
    vw = v[2] * W
    vh = vw * FH / FW
    x0, y0 = v[0] * W - vw / 2, v[1] * H - vh / 2
    sc = FW / vw
    fr = BG.copy()
    # paper position on the frame
    px0, py0, px1, py1 = -x0 * sc, -y0 * sc, (W - x0) * sc, (H - y0) * sc
    if px0 > -40 or py0 > -40 or px1 < FW + 40 or py1 < FH + 40:       # paper edge visible: soft shadow
        sh = Image.new("L", (FW, FH), 0)
        ImageDraw.Draw(sh).rectangle([px0 + 6, py0 + 14, px1 + 6, py1 + 14], fill=150)
        sh = sh.filter(ImageFilter.GaussianBlur(18))
        fr.paste((4, 8, 18), (0, 0), sh)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + vw), min(H, y0 + vh)
    crop = im.crop((round(cx0), round(cy0), round(cx1), round(cy1)))
    ow, oh = max(1, round((cx1 - cx0) * sc)), max(1, round((cy1 - cy0) * sc))
    crop = crop.resize((ow, oh), Image.LANCZOS if sc < 1 else Image.BICUBIC, reducing_gap=3.0 if sc < 0.5 else None)
    fr.paste(crop.convert("RGB"), (round((cx0 - x0) * sc), round((cy0 - y0) * sc)))
    return fr


def rrect(d, box, r, fill):
    d.rounded_rectangle(box, r, fill=fill)


def caption(fr, title, sub, tag, alpha):
    if alpha <= 0:
        return fr
    ov = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    ft, fs, fg = F(True, 40), F(False, 27), F(True, 20)
    w = max(d.textlength(title, font=ft), d.textlength(sub, font=fs)) + 72
    x, y = 56, FH - 196
    rrect(d, (x, y, x + w, y + 140), 16, (12, 22, 44, 228))
    d.rectangle((x, y + 18, x + 6, y + 122), fill=GOLD + (255,))
    d.text((x + 34, y + 22), title, font=ft, fill=WHITE + (255,))
    d.text((x + 34, y + 80), sub, font=fs, fill=MUTED + (255,))
    tw = d.textlength(tag, font=fg) + 36
    rrect(d, (FW - 56 - tw, 48, FW - 56, 92), 22, (12, 22, 44, 228))
    d.text((FW - 56 - tw + 18, 59), tag, font=fg, fill=GOLD + (255,))
    a = ov.split()[3].point(lambda p: round(p * alpha))
    ov.putalpha(a)
    fr = fr.convert("RGBA")
    fr.alpha_composite(ov)
    return fr.convert("RGB")


def fit(d, text, bold, size, maxw):
    """the font at size, or smaller until the text fits maxw"""
    while size > 14 and d.textlength(text, font=F(bold, size)) > maxw:
        size -= 2
    return F(bold, size)


def centred(d, y, text, font, fill):
    d.text(((FW - d.textlength(text, font=font)) / 2, y), text, font=font, fill=fill)


LOGO = None


def title_card(t):
    fr = BG.copy()
    d = ImageDraw.Draw(fr)
    k = ease(t / 0.8)
    lg = LOGO.resize((240, 240), Image.LANCZOS)
    a = lg.split()[3].point(lambda p: round(p * k))
    fr.paste(lg, ((FW - 240) // 2, 230 - round(20 * (1 - k))), a)
    ov = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
    o = ImageDraw.Draw(ov)
    k2 = ease((t - 0.35) / 0.8)
    centred(o, 515, CFG["name"], F(True, 76), WHITE + (round(255 * k2),))
    k3 = ease((t - 0.8) / 0.8)
    centred(o, 625, CFG["tagline"], F(False, 38), GOLD + (round(255 * k3),))
    centred(o, 688, CFG["info"],
            F(False, 28), MUTED + (round(255 * k3),))
    fr = fr.convert("RGBA")
    fr.alpha_composite(ov)
    return fr.convert("RGB")


THUMBS = None


def wall(t):
    fr = BG.copy()
    cols, rows, cw, ch = CFG.get("grid", (8, 6, 214, 139))
    gx, gy = (FW - cols * cw - (cols - 1) * 14) // 2, 110
    for i, th in enumerate(THUMBS[: cols * rows]):
        k = ease((t - i * 0.03) / 0.35)
        if k <= 0:
            continue
        x, y = gx + (i % cols) * (cw + 14), gy + (i // cols) * (ch + 14) + round(18 * (1 - k))
        im = th if k >= 1 else Image.blend(Image.new("RGB", th.size, NAVY1), th, k)
        fr.paste(im, (x, y))
    ov = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
    o = ImageDraw.Draw(ov)
    k = ease((t - 1.6) / 0.6)
    if k > 0:
        o.rounded_rectangle((330, 400, FW - 330, 690), 22, fill=(12, 22, 44, round(238 * k)))
        centred(o, 440, CFG["wall"][0], fit(o, CFG["wall"][0], True, 66, FW - 760), WHITE + (round(255 * k),))
        centred(o, 535, CFG["wall"][1], fit(o, CFG["wall"][1], False, 32, FW - 760), GOLD + (round(255 * k),))
        centred(o, 592, CFG["wall"][2], fit(o, CFG["wall"][2], False, 28, FW - 760), MUTED + (round(255 * k),))
    fr = fr.convert("RGBA")
    fr.alpha_composite(ov)
    return fr.convert("RGB")


def end_card(t):
    fr = BG.copy()
    k = ease(t / 0.7)
    lg = LOGO.resize((170, 170), Image.LANCZOS)
    fr.paste(lg, ((FW - 170) // 2, 200), lg.split()[3].point(lambda p: round(p * k)))
    ov = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
    o = ImageDraw.Draw(ov)
    a = lambda d: round(255 * ease((t - d) / 0.7))
    centred(o, 410, CFG["name"], F(True, 64), WHITE + (a(0.2),))
    centred(o, 520, "Created by Sunny Chase", F(True, 44), GOLD + (a(0.6),))
    centred(o, 582, "github.com/sunnychase", F(False, 32), MUTED + (a(0.6),))
    centred(o, 640, "in collaboration with Claude (Anthropic)", F(False, 32), MUTED + (a(0.9),))
    centred(o, 770, "github.com/sunnychase/" + CFG["repo"], F(False, 30), WHITE + (a(1.3),))
    centred(o, 822, CFG["foot"], F(False, 24), MUTED + (a(1.3),))
    fr = fr.convert("RGBA")
    fr.alpha_composite(ov)
    return fr.convert("RGB")


def timeline():
    """list of (name, start, length, render(t)) pieces; consecutive pieces overlap by FADE."""
    pieces, t = [], 0.0
    pieces.append(("title", t, TITLE_S)); t += TITLE_S - FADE
    for i, s in enumerate(SEGS):
        L = seg_len(s[4]) + FADE
        pieces.append((f"seg{i}", t, L)); t += L - FADE
    pieces.append(("wall", t, WALL_S)); t += WALL_S - FADE
    pieces.append(("end", t, END_S)); t += END_S
    return pieces, t


def render_piece(name, lt):
    if name == "title":
        return title_card(lt)
    if name == "wall":
        return wall(lt)
    if name == "end":
        return end_card(lt)
    s = SEGS[int(name[3:])]
    L = seg_len(s[4]) + FADE
    fr = camera(sheet(s[0]), view_at(s[4], lt))
    ca = ease((lt - 0.35) / 0.45) * (1 - ease((lt - (L - 0.55)) / 0.4))
    return caption(fr, s[1], s[2], s[3], ca)


def init():
    global BG, LOGO, THUMBS
    BG = bg()
    LOGO = Image.open(os.path.join(HERE, "logo.png")).convert("RGBA")
    cw, ch = CFG.get("grid", (8, 6, 214, 139))[2:]
    THUMBS = []
    for p in sorted(glob.glob(os.path.join(HERE, "png", "*.png"))):       # each page fitted into its cell, never stretched
        im = Image.open(p).convert("RGB")
        k = min(cw / im.width, ch / im.height)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        cell = Image.new("RGB", (cw, ch), NAVY1)
        cell.paste(im, ((cw - im.width) // 2, (ch - im.height) // 2))
        THUMBS.append(cell)


def do_frame(args):
    n, t, pieces = args
    act = [(nm, t - st, L) for nm, st, L in pieces if st <= t < st + L]
    if len(act) == 1:
        fr = render_piece(act[0][0], act[0][1])
    else:
        (a, ta, _), (b, tb, _) = act[0], act[-1]
        fr = Image.blend(render_piece(a, ta), render_piece(b, tb), ease(tb / FADE))
    if t < 0.4:
        fr = Image.blend(Image.new("RGB", fr.size, NAVY0), fr, ease(t / 0.4))
    fr.save(os.path.join(OUT, f"f{n:05d}.png"), compress_level=1)
    return n


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pieces, total = timeline()
    N = round(total * FPS)
    print(f"{total:.1f} s, {N} frames")
    # group frames by piece so each worker keeps one big sheet in memory at a time
    jobs = [(n, n / FPS, pieces) for n in range(N)]
    with Pool(int(os.environ.get("REEL_PROCS", "4")), initializer=init) as p:
        for i, _ in enumerate(p.imap(do_frame, jobs, chunksize=12)):
            if i % 150 == 0:
                print(i, flush=True)
