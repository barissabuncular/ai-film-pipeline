"""Place one-shot samples on spotted frames (sync sound / Foley placement).

Spotting CSV columns:  time,sample,gain_db,pitch
  time     seconds from the start of the SHOT (not the cut)
  sample   a wav file, or a folder of wavs to pick from at random (round-robin, no immediate repeats)
  gain_db  optional per-event gain (default 0); small random +-1.5 dB variation is added
  pitch    optional semitone shift (default 0); e.g. -5 for slow motion. Small random +-0.5 st variation is added

Usage:
  python tools/foley.py sound/spot_mp08.csv --out sound/foley_mp08.wav [--offset 39.7] [--length 5.06] [--seed 3]
--offset shifts every event by the shot's start time in the cut, so the output can go straight into cues.csv at start 0.
Without --offset the track starts at the shot start (use the shot start as the cue start).
"""
import argparse, csv, os, random, subprocess
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(sample_path):
    p = sample_path if os.path.isabs(sample_path) else os.path.join(HERE, sample_path)
    if os.path.isdir(p):
        return sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".wav"))
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--length", type=float, default=None, help="pad/trim the output to this many seconds (after offset)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="random timing jitter per event (0 = frame-exact)")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    events = []
    with open(a.spot, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("time"):
                continue
            events.append((float(r["time"]), r["sample"].strip(), float(r.get("gain_db") or 0), float(r.get("pitch") or 0)))
    if not events:
        raise SystemExit("no events")

    cmd = [FFMPEG, "-y", "-loglevel", "error"]
    fc, labels = [], []
    last = {}
    for i, (t, sample, gain, pitch) in enumerate(events):
        pool = resolve(sample)
        choice = rng.choice([p for p in pool if p != last.get(sample)] or pool)
        last[sample] = choice
        cmd += ["-i", choice]
        g = gain + rng.uniform(-1.5, 1.5)
        st = pitch + rng.uniform(-0.5, 0.5)
        ratio = 2 ** (st / 12.0)
        ms = int(max(0.0, (t + a.offset) * 1000 + rng.uniform(-a.jitter_ms, a.jitter_ms)))
        # pitch shift by resampling (asetrate) then bringing the rate back; changes speed too, which is what Foley pitching does
        fc.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,asetrate=48000*{ratio:.5f},aresample=48000,"
                  f"volume={g:.2f}dB,adelay={ms}|{ms}[e{i}]")
        labels.append(f"[e{i}]")
    fc.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:dropout_transition=0[mix]")
    tail = f",apad=whole_dur={a.length + a.offset:.3f},atrim=0:{a.length + a.offset:.3f}" if a.length else ""
    fc[-1] = fc[-1].replace("[mix]", f"{tail}[mix]") if tail else fc[-1]
    cmd += ["-filter_complex", ";".join(fc), "-map", "[mix]", "-c:a", "pcm_s16le", a.out]
    subprocess.run(cmd, check=True)
    print(f"{a.out}: {len(events)} events placed (offset {a.offset}s)")


if __name__ == "__main__":
    main()
