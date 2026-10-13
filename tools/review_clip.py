"""Make a contact sheet (grid of evenly spaced frames) from one or more mp4 clips so they can be judged as images.

Usage: python tools/review_clip.py <clip.mp4> [more.mp4 ...] [--frames 8] [--cols 4] [--out sheet.png]
With several clips, one row per clip is produced in a single sheet (out defaults to next to the first clip).
"""
import argparse, os, subprocess, tempfile
import imageio_ffmpeg
from PIL import Image, ImageDraw

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path):
    out = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 5.0


def extract(path, n, tmp):
    dur = probe_duration(path)
    frames = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        f = os.path.join(tmp, f"f{i:02d}.png")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", f], check=True)
        frames.append(Image.open(f).convert("RGB"))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=416, help="thumbnail width")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in a.clips:
            fr = extract(c, a.frames, tmp)
            rows.append((os.path.basename(c), fr))
    tw = a.thumb
    th = int(tw * rows[0][1][0].height / rows[0][1][0].width)
    per_row = -(-a.frames // a.cols)
    sheet_h = sum(per_row * th + 18 for _ in rows)
    sheet = Image.new("RGB", (a.cols * tw, sheet_h), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    y = 0
    for name, fr in rows:
        d.text((4, y + 2), name, fill=(230, 230, 230))
        y += 18
        for i, im in enumerate(fr):
            im = im.resize((tw, th))
            sheet.paste(im, ((i % a.cols) * tw, y + (i // a.cols) * th))
        y += per_row * th
    out = a.out or os.path.splitext(a.clips[0])[0] + "_sheet.png"
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
