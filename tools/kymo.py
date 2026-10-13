"""Kymographs for spotting dense action: for each vertical strip of the frame, an image of height against time.

A falling object draws a diagonal streak; where the streak goes flat is the contact frame. Reading these is faster and
more complete than reading contact sheets when many objects move at once (casings, debris, rain on a surface), and it
catches every fall because nothing can cross a strip without leaving a trace.

Works on a frame dump made by tools/spot.py (f001.jpg/png ...). Frame index N (0-based, as on spot.py's sheets) is
column N; the Wan chunk frames (index 1, 9, 17 ... at 32 fps, i.e. every 4 source frames) are marked with blue lines,
since the whole picture jumps slightly there and a jump is not a contact.

Usage:
  python tools/kymo.py review/spot_<id> --out review/kymo_<id> [--x 350,640] [--y 130,500] [--band 30] [--per 4]
                       [--mode dark|bright] [--scale 6] [--chunk 8 --chunk-offset 1]
  --x / --y    pixel range of the dumped frames to analyse (default: whole frame)
  --band       strip width in pixels; each strip is one panel
  --mode       dark: min over the strip (dark objects on a bright floor); bright: max (lit objects, sparks, brass)
  --per        panels per output image
Writes <out>/ky_<x>.jpg images; frame numbers are printed along the top in red.
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--x", default=None)
    ap.add_argument("--y", default=None)
    ap.add_argument("--band", type=int, default=30)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--mode", choices=["dark", "bright"], default="bright")
    ap.add_argument("--scale", type=int, default=6, help="pixels per frame along time")
    ap.add_argument("--chunk", type=int, default=8, help="chunk period in frames (8 at 32 fps, 4 at 16 fps, 0 = off)")
    ap.add_argument("--chunk-offset", type=int, default=1)
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.frames) if f.startswith("f") and f[1:4].isdigit() and f.endswith((".jpg", ".png")))
    if not files:
        raise SystemExit(f"no f###.jpg/png frames in {a.frames} (make them with tools/spot.py)")
    fr = np.stack([np.asarray(Image.open(os.path.join(a.frames, f)).convert("L"), dtype=np.float32) for f in files])
    N, H, W = fr.shape
    x0, x1 = (int(v) for v in a.x.split(",")) if a.x else (0, W)
    y0, y1 = (int(v) for v in a.y.split(",")) if a.y else (0, H)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    TS = a.scale

    def panel(bx):
        band = fr[:, y0:y1, bx:bx + a.band]
        img = (band.min(axis=2) if a.mode == "dark" else band.max(axis=2)).T  # rows = height, cols = time
        lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
        img = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(img).resize((N * TS, y1 - y0), Image.NEAREST).convert("RGB")
        d = ImageDraw.Draw(im)
        if a.chunk:
            for n in range(a.chunk_offset, N, a.chunk):
                d.line([(n * TS, 0), (n * TS, y1 - y0)], fill=(0, 90, 170))
        for n in range(0, N, 2):
            d.line([(n * TS, 0), (n * TS, 10 if n % 8 == 0 else 3)], fill=(255, 0, 0))
        for n in range(0, N, 8):
            d.text((n * TS + 2, 11), str(n), fill=(255, 70, 70), font=font)
        for yy in range(y0 - y0 % 50 + 50, y1, 50):
            d.text((2, yy - y0 - 7), str(yy), fill=(0, 255, 0), font=font)
        lab = Image.new("RGB", (im.width, 18), (0, 0, 0))
        ImageDraw.Draw(lab).text((4, 1), f"strip x {bx}-{bx + a.band} of {W} px, y {y0}-{y1}; red = frame index, blue = chunk frames",
                                 fill=(255, 255, 0), font=font)
        p = Image.new("RGB", (im.width, im.height + 18))
        p.paste(lab, (0, 0)); p.paste(im, (0, 18))
        return p

    os.makedirs(a.out, exist_ok=True)
    xs = list(range(x0, x1, a.band))
    for i in range(0, len(xs), a.per):
        ps = [panel(x) for x in xs[i:i + a.per]]
        S = Image.new("RGB", (ps[0].width, sum(p.height for p in ps)))
        y = 0
        for p in ps:
            S.paste(p, (0, y)); y += p.height
        name = f"ky_{xs[i]:04d}.jpg"
        S.save(os.path.join(a.out, name), quality=90)
        print(name, S.size)


if __name__ == "__main__":
    main()
