"""Sync-check video: the picture over a DAW-style scrolling waveform with a centre playhead and a labelled line at
every spotted event. This is how to judge sync by eye instead of by feel.

Reading it: the red playhead is "now". Each coloured line on the strip is a spotted frame (yellow = hard effect,
cyan = foley, magenta = design); its label names the shot, frame and event, and the corner box flashes on that frame.
In sync = the picture event, the line and the waveform spike all reach the playhead together. A spike left of its line
means the sound is early (worst: audiences notice sound leading picture from about 45 ms), right of it means late
(tolerated up to about 1 frame, sound travels slower than light). The strip shows 1 frame = pps/fps pixels.

Events come from the spotting files (sound/spotting/<id>.csv: time from the SHOT start) plus extra per-shot CSVs with a
time column (e.g. a casings shot, a footsteps shot). Shot starts in the cut are computed from the edit list (dissolves included).

Usage:
  python tools/synccheck.py --video output/fx_only.mp4 --out output/synccheck.mp4
         [--audio sound/mix/fx_only.wav] [--edit edit.txt --fade 0.4] [--spotting sound/spotting]
         [--extra s08=sound/spot_s08.csv,s06=sound/spot_s06.csv] [--pps 400] [--slow 0.5]
         [--wave sound/mix/hits_only.wav]
--wave draws the strip from a different file than the one you hear: give it the sync stems alone (no beds, music or
voice) so every spike on the strip is one placed sound; otherwise continuous beds fill the strip.
--slow 0.5 renders at half speed (audio pitch kept) for frame-by-frame inspection; 1.0 is the real-speed test.
"""
import argparse, csv, glob, os, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "C:/Windows/Fonts/consola.ttf"
KIND_COLOUR = {"hard": (255, 214, 0), "foley": (0, 220, 255), "design": (255, 80, 220)}


def probe_duration(path):
    out = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"no duration for {path}")


def shot_starts(edit, fade, fps=32):
    """Shot id -> (start in the cut, duration used), from an edit list of path|start|duration lines.
    The joiner (tools/assemble.py, ffmpeg xfade) puts each shot's frame 0 on the output frame grid, i.e. at
    round(nominal_start * fps) / fps, up to half a frame away from the nominal start. Events are timed from that."""
    starts, t = {}, 0.0
    for line in open(os.path.join(HERE, edit), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        path = parts[0].strip()
        sid = os.path.basename(os.path.dirname(path))
        full = probe_duration(os.path.join(HERE, path))
        s0 = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        dur = min(float(parts[2]) if len(parts) > 2 and parts[2] else full - s0, full - s0)
        starts[sid] = (round(t * fps) / fps, dur)
        t += dur - fade
    return starts


def clean(s, n=34):
    return re.sub(r"[^A-Za-z0-9 .,+\-/()]", "", s)[:n]


def load_events(spotting_dir, extra, starts):
    ev = []
    for p in sorted(glob.glob(os.path.join(HERE, spotting_dir, "*.csv"))):
        sid = os.path.splitext(os.path.basename(p))[0]
        if sid not in starts:
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            kind = (r.get("kind") or "foley").strip()
            if not r.get("time") or kind not in KIND_COLOUR:  # beds and movement rows have no sync point
                continue
            t = float(r["time"])
            if t > starts[sid][1]:
                continue
            fr = r.get("frame") or str(round(t * 32))
            ev.append((starts[sid][0] + t, kind, f"{sid} f{fr} {clean(r.get('event') or r.get('sound') or '', 22)}"))
    for item in filter(None, (extra or "").split(",")):
        sid, p = item.split("=", 1)
        for r in csv.DictReader(open(os.path.join(HERE, p), encoding="utf-8")):
            if not r.get("time"):
                continue
            t = float(r["time"])
            fr = r.get("frame") or str(round(t * 32))
            ev.append((starts[sid][0] + t, "foley", f"{sid} f{fr} {clean(r.get('evidence') or r.get('sample') or '', 16)}"))
    return sorted(ev)


def read_audio(path, sr=8000):
    raw = subprocess.run([FFMPEG, "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def draw_strip(x, sr, dur, events, starts, pps, h, path):
    W = int(dur * pps) + 2
    im = Image.new("RGB", (W, h), (16, 16, 20))
    d = ImageDraw.Draw(im)
    f_small = ImageFont.truetype(FONT, 13)
    f_shot = ImageFont.truetype(FONT, 16)
    wave_top, mid = 70, 70 + (h - 70) // 2
    half = (h - 70) // 2 - 4
    # frame ticks every 1/32 s
    for n in range(int(dur * 32) + 1):
        xx = int(n / 32 * pps)
        d.line([(xx, h - 6), (xx, h)], fill=(70, 70, 80))
    for sid, (st, du) in starts.items():
        xx = int(st * pps)
        d.line([(xx, wave_top - 8), (xx, h)], fill=(90, 90, 110), width=2)
        d.text((xx + 4, wave_top - 8), sid, fill=(150, 150, 170), font=f_shot)
    # waveform: per-column peak, on a dB scale so quiet foley stays visible next to thunder
    spp = sr / pps
    for c in range(W):
        seg = x[int(c * spp):int((c + 1) * spp)]
        if not len(seg):
            break
        pk = float(np.abs(seg).max())
        db = 20 * np.log10(pk + 1e-9)
        a = max(0.0, min(1.0, (db + 60) / 60))
        if a > 0:
            d.line([(c, mid - a * half), (c, mid + a * half)], fill=(120, 200, 120))
    for i, (t, kind, label) in enumerate(events):
        xx = int(t * pps)
        col = KIND_COLOUR.get(kind, (255, 255, 255))
        d.line([(xx, 2 + (i % 4) * 16 + 14), (xx, h)], fill=col, width=1)
        d.text((xx + 2, 2 + (i % 4) * 16), label, fill=col, font=f_small)
    im.save(path)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=None, help="audio to show and play (default: the video's own audio)")
    ap.add_argument("--edit", default="edit.txt")
    ap.add_argument("--fade", type=float, default=0.4)
    ap.add_argument("--spotting", default="sound/spotting")
    ap.add_argument("--extra", default="")
    ap.add_argument("--pps", type=int, default=400, help="strip pixels per second (400 = 12.5 px per 32 fps frame)")
    ap.add_argument("--strip-h", type=int, default=260)
    ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--wave", default=None, help="audio drawn on the strip (default: the audio you hear)")
    a = ap.parse_args()

    starts = shot_starts(a.edit, a.fade)
    events = load_events(a.spotting, a.extra, starts)
    audio = a.audio or a.video
    x, sr = read_audio(a.wave or audio)
    dur = probe_duration(a.video)
    work = os.path.splitext(a.out)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    strip = os.path.join(work, "strip.png")
    draw_strip(x, sr, dur, events, starts, a.pps, a.strip_h, strip)
    print(f"{len(events)} events, strip {int(dur * a.pps)} px")

    fontfile = FONT.replace(":", "\\:")
    H = 1080 + a.strip_h
    fc = [f"[0:v]scale=1920:1080,setsar=1,pad=1920:{H}:0:0:black[base]",
          f"[1:v]format=rgb24[strip]",
          f"[base][strip]overlay=x='960-t*{a.pps}':y=1080:shortest=1:eval=frame[v0]",
          f"[v0]drawbox=x=959:y=1080:w=3:h={a.strip_h}:color=red@1:t=fill[v1]"]
    last = "v1"
    # current shot and frame, top left
    for i, (sid, (st, du)) in enumerate(starts.items()):
        y = 20 + (i % 2) * 48
        txt = f"{sid}  f%{{eif\\:(t-{st:.3f})*32\\:d}}"
        fc.append(f"[{last}]drawtext=fontfile='{fontfile}':text='{txt}':x=24:y={y}:fontsize=40:fontcolor=white:"
                  f"box=1:boxcolor=black@0.6:enable='between(t,{st:.3f},{st + du:.3f})'[s{i}]")
        last = f"s{i}"
    # corner flash on every event frame (2 frames long), coloured by kind
    for i, (t, kind, label) in enumerate(events):
        col = "0x%02x%02x%02x" % KIND_COLOUR.get(kind, (255, 255, 255))
        fc.append(f"[{last}]drawbox=x=1780:y=20:w=120:h=120:color={col}@1:t=fill:enable='between(t,{t:.4f},{t + 2 / 32:.4f})'[e{i}]")
        last = f"e{i}"
    vout = last
    if a.slow != 1.0:
        fc.append(f"[{last}]setpts=PTS/{a.slow}[vs]")
        vout = "vs"
        tempo, chain = a.slow, []
        while tempo < 0.5:
            chain.append("atempo=0.5"); tempo /= 0.5
        chain.append(f"atempo={tempo:.4f}")
        fc.append(f"[{2 if a.audio else 0}:a]{','.join(chain)}[aout]")
        amap = "[aout]"
    else:
        amap = f"{2 if a.audio else 0}:a"
    script = os.path.join(work, "filter.txt")
    open(script, "w", encoding="utf-8").write(";\n".join(fc))
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", a.video, "-loop", "1", "-framerate", "32", "-i", strip]
    if a.audio:
        cmd += ["-i", a.audio]
    cmd += ["-filter_complex_script", script, "-map", f"[{vout}]", "-map", amap,
            "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
