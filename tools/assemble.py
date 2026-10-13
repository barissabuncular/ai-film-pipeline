"""Assemble an ordered list of clips into one trailer mp4 with crossfades, scaled to 1920x1080.

Usage: python tools/assemble.py edit.txt --out trailer.mp4 [--fade 0.4] [--fps 24]
edit.txt: one clip path per line (relative to project root allowed), '#' comments.
Optional per-line trim: "path|start|duration" in seconds.
"""
import argparse, os, subprocess
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def probe_duration(path):
    out = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"no duration for {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edit")
    ap.add_argument("--out", default="trailer.mp4")
    ap.add_argument("--fade", type=float, default=0.4)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--tail", type=float, default=0.0, help="seconds of black held after the last clip (for a final line or title)")
    a = ap.parse_args()

    items = []
    with open(a.edit, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            path = parts[0].strip()
            if not os.path.isabs(path):
                path = os.path.join(HERE, path)
            start = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
            full = probe_duration(path)
            dur = float(parts[2]) if len(parts) > 2 and parts[2] else full - start
            items.append((path, start, min(dur, full - start)))

    cmd = [FFMPEG, "-y", "-loglevel", "error"]
    for p, s, d in items:
        cmd += ["-ss", f"{s:.3f}", "-t", f"{d:.3f}", "-i", p]

    fc = []
    for i in range(len(items)):
        fc.append(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps={a.fps},setsar=1,format=yuv420p[v{i}]")
    # chain xfades
    prev = "v0"
    offset = 0.0
    for i in range(1, len(items)):
        offset += items[i - 1][2] - a.fade
        out = f"x{i}"
        fc.append(f"[{prev}][v{i}]xfade=transition=fade:duration={a.fade}:offset={offset:.3f}[{out}]")
        prev = out
    total = sum(d for _, _, d in items) - a.fade * (len(items) - 1)
    tail = f",tpad=stop_mode=add:stop_duration={a.tail:.3f}:color=black" if a.tail > 0 else ""
    fc.append(f"[{prev}]fade=t=in:st=0:d=0.8,fade=t=out:st={max(0, total - 1.0):.3f}:d=1.0{tail}[vout]")
    total += a.tail

    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]", "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"{a.out}  {total:.1f}s  {len(items)} clips")


if __name__ == "__main__":
    main()
