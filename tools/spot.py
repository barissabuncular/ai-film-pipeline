"""Spotting helper: dump every frame of a clip with frame numbers, and compute a motion-energy curve
(mean absolute frame difference inside an optional region) so impact frames stand out as peaks.

Usage:
  python tools/spot.py shots/<id>/selected.mp4 --out review/spot_<id> [--region x0,y0,x1,y1 (fractions 0-1)] [--cols 9] [--thumb 200]
                       [--width 640] [--jpg]
--width scales the dumped frames (keeps disk and reading cost low on 1080p finals); --jpg writes f000.jpg instead of png.
Writes: <out>/f000.png ... , <out>/sheet_N.png (numbered contact sheets), <out>/motion.csv (frame,time,energy) and prints peaks.
"""
import argparse, csv, os, subprocess
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path):
    out = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True).stderr
    fps = 16.0
    for line in out.splitlines():
        if "Stream" in line and "Video" in line and "fps" in line:
            for tok in line.split(","):
                if "fps" in tok:
                    fps = float(tok.strip().split()[0])
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", default=None, help="x0,y0,x1,y1 as fractions of width/height")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=210)
    ap.add_argument("--width", type=int, default=0, help="scale dumped frames to this width (0 = native)")
    ap.add_argument("--jpg", action="store_true", help="dump frames as jpg instead of png")
    a = ap.parse_args()
    ext = ".jpg" if a.jpg else ".png"

    os.makedirs(a.out, exist_ok=True)
    fps = probe(a.clip)
    vf = ["-vf", f"scale={a.width}:-2"] if a.width else []
    q = ["-q:v", "2"] if a.jpg else []
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", a.clip, *vf, *q, os.path.join(a.out, "f%03d" + ext)], check=True)
    frames = sorted(f for f in os.listdir(a.out) if f.startswith("f") and f.endswith(ext))
    print(f"{len(frames)} frames at {fps:g} fps")

    # motion energy
    region = [float(v) for v in a.region.split(",")] if a.region else None
    prev = None
    rows = []
    for i, f in enumerate(frames):
        im = Image.open(os.path.join(a.out, f)).convert("L")
        if region:
            W, H = im.size
            im = im.crop((int(region[0] * W), int(region[1] * H), int(region[2] * W), int(region[3] * H)))
        arr = np.asarray(im, dtype=np.float32)
        e = float(np.abs(arr - prev).mean()) if prev is not None else 0.0
        rows.append((i, i / fps, e))
        prev = arr
    with open(os.path.join(a.out, "motion.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "time", "energy"]); w.writerows(rows)
    en = np.array([r[2] for r in rows])
    if len(en) > 3:
        thr = en.mean() + 0.8 * en.std()
        peaks = [i for i in range(1, len(en) - 1) if en[i] > thr and en[i] >= en[i - 1] and en[i] >= en[i + 1]]
        print("motion peaks (frame @ time):", ", ".join(f"{p}@{p / fps:.2f}s" for p in peaks))

    # numbered contact sheets
    per = a.cols * a.rows
    tw = a.thumb
    im0 = Image.open(os.path.join(a.out, frames[0]))
    th = int(tw * im0.height / im0.width)
    for s in range(0, len(frames), per):
        chunk = frames[s:s + per]
        sheet = Image.new("RGB", (a.cols * tw, -(-len(chunk) // a.cols) * (th + 14)), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for j, f in enumerate(chunk):
            im = Image.open(os.path.join(a.out, f)).convert("RGB")
            if region:
                W, H = im.size
                im = im.crop((int(region[0] * W), int(region[1] * H), int(region[2] * W), int(region[3] * H)))
                th_r = int(tw * im.height / im.width)
                im = im.resize((tw, th_r))
            else:
                im = im.resize((tw, th))
            x = (j % a.cols) * tw; y = (j // a.cols) * (th + 14)
            sheet.paste(im, (x, y + 14))
            idx = s + j
            d.text((x + 3, y + 1), f"{idx} {idx / fps:.2f}s", fill=(240, 240, 240))
        sheet.save(os.path.join(a.out, f"sheet_{s // per + 1}.png"))
    print("sheets:", -(-len(frames) // per))


if __name__ == "__main__":
    main()
