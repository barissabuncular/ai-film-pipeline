"""Turn generated or library clips into transient-aligned one-shots.

For each input wav: find the first onset (where the short-window RMS jumps above a threshold relative to the peak),
cut from 5 ms before it, keep --keep seconds, fade out the last 60 ms, normalise the peak to -3 dBFS, and write to --out.
Clips whose peak is below --min-peak dBFS (no real hit) are skipped.

--single   one event, one sound, for the sample too. After aligning to the main hit, find the first SECONDARY onset starting
           more than --min-gap s after the main hit's onset whose envelope rises by --rise-db (within 20 ms) to above
           --second-db dB relative to the main hit's peak (the loudest point of the first 50 ms): a bounce, a rattle, a
           second footfall. Cut just before it with a --cut-fade s fade that ends where that onset starts to rise, so all
           the natural ring up to it is kept. Clips whose remaining length is under --min-len s are skipped. Without a
           secondary onset the clip is cut by --keep as usual.
           The onset envelope is the Hilbert magnitude of the DC-free mono sum, smoothed over --env-ms ms: 5 ms for clicks
           and metal; 10-20 ms for crunchy or textured sounds (snow, cloth, paper), whose micro-texture is not an event;
           40 ms or more for sub-heavy booms, whose low partials beat. Off by default, so old commands give the same files.
--count    analyse only: print the events per input (onsets as above that start more than --min-gap after the previous
           event's onset, the rule --single cuts by) and how many files hold exactly one; nothing is written and --out is
           not needed. Use it before and after --single, and with a lower --second-db (e.g. -24) to see the margin.
--sr       resample the output (e.g. 48000; default keeps the source rate). --mono folds to one channel: the L+R mean,
           or the louder channel when L/R are poorly correlated (spaced mics), which would comb-filter in a sum; the
           detection then runs on that folded channel. --hp HZ high-passes the input first (2nd order), so sub-audio
           drift or rumble neither passes the --min-peak gate nor steers the alignment (casings: --hp 40).

As a module: trim(x, sr, **options) trims one clip exactly like the CLI (options named like the flags, sr_out for
--sr) and count_events(mono, sr, ...) is the --count rule; a pool-building script can import both.

Usage: python tools/oneshot_trim.py sound/os_casing_*.wav --out sound/oneshots/casing --keep 0.6
       python tools/oneshot_trim.py "sound/lib/casing/*.wav" --out sound/oneshots/pool/casing --single --hp 40 --sr 48000 --mono
       python tools/oneshot_trim.py "sound/oneshots/pool/casing/*.wav" --count
Needs numpy, scipy, soundfile (the audio venv, AUDIO_ROOT\\.venv-ace).
"""
import argparse, glob, os
from math import gcd
import numpy as np, soundfile as sf
from scipy.signal import butter, hilbert, resample_poly, sosfilt


def envelope_db(mono, sr, env_ms=5.0):
    """Hilbert envelope of the DC-free signal, moving-average smoothed over env_ms, sampled every 1 ms, in dB."""
    x = np.asarray(mono, dtype=np.float64)
    x = x - x.mean()
    e = np.abs(hilbert(x))
    w = max(1, int(sr * env_ms / 1000))
    e = np.convolve(e, np.ones(w) / w, mode="same")
    hop = max(1, int(sr * 0.001))
    return 20 * np.log10(e[::hop] + 1e-12), hop


def find_onsets(mono, sr, thr_db=-18.0, rise_db=6.0, env_ms=5.0, ref_db=None, look_ms=20, refr_ms=30):
    """Onsets where the envelope rises by >= rise_db (vs its minimum over the previous look_ms) to >= thr_db re ref_db
    (default: the envelope maximum). Returns [(rise_start_s, detect_s, level_db, rise_db)]. Silence is assumed before
    the file, so a hit at the very start counts as an onset."""
    e, hop = envelope_db(mono, sr, env_ms)
    e = e - (e.max() if ref_db is None else ref_db)
    ep = np.concatenate([np.full(look_ms, -120.0), e])
    out, last = [], -10**9
    for k in range(len(e)):
        seg = ep[k:k + look_ms]
        lo = seg.min()
        if e[k] >= thr_db and e[k] - lo >= rise_db and k - last >= refr_ms:
            st = max(0, k - look_ms + int(np.argmin(seg)))
            out.append((st * hop / sr, k * hop / sr, float(e[k]), float(e[k] - lo)))
            last = k
    return out


def count_events(mono, sr, thr_db=-18.0, rise_db=6.0, env_ms=5.0, min_gap=0.06):
    """Distinct events: onsets (find_onsets) that start more than min_gap s after the previous event's onset, the same
    rule --single cuts by. Returns [(onset_s, level_db)]."""
    ev = []
    for st, det, lvl, r in find_onsets(mono, sr, thr_db, rise_db, env_ms):
        if not ev or st > ev[-1][0] + min_gap:
            ev.append((st, lvl))
    return ev


def fold_mono(x):
    if x.shape[1] == 1:
        return x
    c = np.corrcoef(x[:, 0], x[:, 1])[0, 1] if x[:, 0].std() > 0 and x[:, 1].std() > 0 else 1.0
    if c >= 0.5:
        return x.mean(axis=1, keepdims=True)
    return x[:, [int(np.argmax((x ** 2).sum(axis=0)))]]


def resample(y, sr, sr_out):
    if not sr_out or sr_out == sr:
        return y, sr
    g = gcd(int(sr), int(sr_out))
    return resample_poly(y, int(sr_out) // g, int(sr) // g, axis=0).astype(np.float32), int(sr_out)


def files_from(inputs):
    files = []
    for p in inputs:
        files += glob.glob(p) if any(c in p for c in "*?[") else [p]
    return sorted(files)


DEFAULTS = dict(keep=0.6, min_peak=-30.0, onset_ratio=0.25, align="peak", min_front=0.6, single=False, min_gap=0.06,
                second_db=-18.0, rise_db=6.0, env_ms=5.0, cut_fade=0.02, min_len=0.1, sr_out=0, mono=False, hp=0.0)


def trim(x, sr, **kw):
    """Trim one clip (frames x channels) as the CLI does. Returns (y, sr_out, info) or (None, None, reason).
    info: start_s (cut start in the source), len_s, cut_s / cut_db (secondary onset, None if cut by --keep)."""
    a = dict(DEFAULTS, **kw)
    if a["hp"]:
        x = sosfilt(butter(2, a["hp"], "hp", fs=sr, output="sos"), x, axis=0)
    if a["mono"]:
        x = fold_mono(x)  # detect on exactly what will be written
    mono = x.mean(axis=1)
    peak = np.abs(mono).max()
    if peak <= 0 or 20 * np.log10(peak) < a["min_peak"]:
        return None, None, "quiet"
    if a["align"] == "peak":
        idx = int(np.argmax(np.abs(mono)))
        start = max(0, idx - int(sr * 0.008))
    else:
        win = max(1, int(sr * 0.005))
        n = len(mono) // win
        rms = np.sqrt((mono[: n * win].reshape(n, win) ** 2).mean(axis=1))
        thr = a["onset_ratio"] * rms.max()
        start = max(0, int(np.argmax(rms > thr)) * win - int(sr * 0.005))
    end = min(len(x), start + int(sr * a["keep"]))
    info = dict(start_s=start / sr, cut_s=None, cut_db=None)
    fade_s = 0.06
    if a["single"]:
        seg = mono[start:end]
        e, hop = envelope_db(seg, sr, a["env_ms"])
        ref = e[:50].max()  # the main hit: loudest envelope point in the first 50 ms after the cut start
        ons = find_onsets(seg, sr, a["second_db"], a["rise_db"], a["env_ms"], ref_db=ref)
        t_main = ons[0][0] if ons else 0.0  # its onset
        sec = [o for o in ons[1:] if o[0] > t_main + a["min_gap"]]
        if sec:
            end = start + int(sec[0][0] * sr)
            fade_s = a["cut_fade"]
            info.update(cut_s=sec[0][0], cut_db=sec[0][2])
        if (end - start) / sr < a["min_len"]:
            return None, None, f"under --min-len after single cut ({(end - start) / sr:.3f}s)"
    y = x[start:end].astype(np.float32)
    y, sr_out = resample(y, sr, a["sr_out"])
    fi = min(len(y), int(sr_out * 0.003))
    if fi > 0:
        y[:fi] *= np.linspace(0, 1, fi)[:, None]
    fade = min(len(y), int(sr_out * fade_s))
    if fade > 0:
        y[-fade:] *= np.linspace(1, 0, fade)[:, None]
    y *= (10 ** (-3 / 20)) / (np.abs(y).max() + 1e-9)
    # keep only front-loaded hits: most energy inside the first 80 ms
    mm = np.abs(y.mean(axis=1))
    e1 = (mm[: int(sr_out * 0.08)] ** 2).sum(); e2 = (mm[int(sr_out * 0.08):] ** 2).sum() + 1e-9
    if e1 / e2 < a["min_front"]:
        return None, None, f"not a single hit: front/rest={e1 / e2:.2f}"
    info["len_s"] = len(y) / sr_out
    return y, sr_out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=None, help="output folder (required unless --count)")
    ap.add_argument("--keep", type=float, default=0.6)
    ap.add_argument("--min-peak", type=float, default=-30.0, help="dBFS; skip clips quieter than this")
    ap.add_argument("--onset-ratio", type=float, default=0.25, help="onset = first window whose RMS exceeds this * max window RMS")
    ap.add_argument("--align", choices=["peak", "onset"], default="peak", help="cut at the loudest transient (default) or the first onset")
    ap.add_argument("--min-front", type=float, default=0.6, help="skip clips whose first 80 ms hold less than this fraction of the rest's energy")
    ap.add_argument("--single", action="store_true", help="cut before the first secondary onset (one event, one sound)")
    ap.add_argument("--min-gap", type=float, default=0.06, help="s after the main hit's onset before a secondary onset can count")
    ap.add_argument("--second-db", type=float, default=-18.0, help="dB re the main hit's peak a secondary onset must rise above")
    ap.add_argument("--rise-db", type=float, default=6.0, help="dB the envelope must rise within 20 ms to count as an onset")
    ap.add_argument("--env-ms", type=float, default=5.0, help="onset envelope smoothing, ms")
    ap.add_argument("--cut-fade", type=float, default=0.02, help="s; fade that ends where the secondary onset starts (15-25 ms)")
    ap.add_argument("--min-len", type=float, default=0.1, help="s; with --single, skip clips shorter than this after the cut")
    ap.add_argument("--count", action="store_true", help="only print events per file, write nothing")
    ap.add_argument("--sr", type=int, default=0, help="output sample rate (0 = keep the source rate)")
    ap.add_argument("--mono", action="store_true", help="fold the output to one channel")
    ap.add_argument("--hp", type=float, default=0.0, help="high-pass the input at this many Hz first (2nd order; 0 = off)")
    a = ap.parse_args()
    files = files_from(a.inputs)

    if a.count:
        one = 0
        for f in files:
            x, sr = sf.read(f, always_2d=True)
            ev = count_events(x.mean(axis=1), sr, a.second_db, a.rise_db, a.env_ms, a.min_gap)
            one += len(ev) == 1
            print(f"{os.path.basename(f)}: {len(ev)} event(s) " + " ".join(f"{t:.3f}s({l:+.0f})" for t, l in ev[:12]))
        print(f"{one}/{len(files)} files hold exactly one event (--second-db {a.second_db}, --min-gap {a.min_gap}, --env-ms {a.env_ms})")
        return
    if not a.out:
        ap.error("--out is required unless --count")

    os.makedirs(a.out, exist_ok=True)
    kw = {k: getattr(a, k) for k in DEFAULTS if k != "sr_out"}
    kw["sr_out"] = a.sr
    kept = 0
    for f in files:
        x, sr = sf.read(f, always_2d=True)
        y, sr_out, info = trim(x, sr, **kw)
        if y is None:
            print(f"skip ({info}): {os.path.basename(f)}"); continue
        name = os.path.splitext(os.path.basename(f))[0] + ".wav"
        sf.write(os.path.join(a.out, name), y, sr_out, subtype="PCM_16")
        kept += 1
        cut = f", cut before secondary onset at {info['cut_s'] * 1000:.0f} ms ({info['cut_db']:+.0f} dB)" if info["cut_s"] is not None else ""
        print(f"{name}: onset {info['start_s'] * 1000:.0f} ms, {info['len_s']:.2f}s{cut}")
    print(f"kept {kept}/{len(files)} -> {a.out}")


if __name__ == "__main__":
    main()
