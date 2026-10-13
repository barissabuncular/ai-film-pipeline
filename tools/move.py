"""Render a sound through movement automation: pan, level, air absorption and doppler follow the source on screen.

A moving source is not a static loop. As it nears it gets louder and brighter (less high end lost in the air), it shifts
pitch while the distance is changing (doppler) and it sits where the picture puts it. This renders that from keyframes.

Automation CSV columns:  time,pan,gain_db,dist[,rate]     ('#' lines are comments; other columns such as note are ignored)
  time     seconds from the start of the SHOT (not the cut). Values are interpolated linearly between rows, held outside them
  pan      -1 hard left .. 0 centre .. +1 hard right, equal-power law (-3 dB each side at centre). For an on-screen source,
           pan = its screen x (tools/track.py writes that)
  gain_db  extra level trim in dB (default 0)
  dist     0 near .. 1 far (perceptual, default 0). Drives: attenuation (--far-db at dist 1, linear in dB),
           a low-pass for air absorption (--near-hz at 0 to --far-hz at 1, log-spaced) and, with --doppler, the pitch
  rate     optional explicit playback rate (1 = as recorded, 1.0595 = +1 semitone); multiplied with the doppler rate
Air absorption is rendered as a crossfade between zero-phase low-passed copies of the sound (--bands of them), so the
tone moves continuously with no zipper noise and no comb filtering. Doppler: dist is mapped to metres
(--near-m .. --far-m), its rate of change is the radial speed v, and the playback rate is c / (c + v) with c = 343 m/s,
smoothed over --doppler-smooth s. Approaching = higher, receding = lower. With the defaults, dist 0.8 -> 0.3 in 3 s is
120 m at 40 m/s = +2.1 semitones while it closes, back to 0 when it stops: audible, not a cartoon. --far-m scales it.

Usage:
  python tools/move.py sound/lib/<source>.wav sound/move_<id>.csv --out sound/move_<id>.wav [--length 5.03] [--offset 53.5]
         [--start 0] [--doppler] [--near-m 10 --far-m 250] [--far-db -15] [--near-hz 16000 --far-hz 2500]
         [--xfade 0.25] [--fade-in 0.01 --fade-out 0.01]
The source is folded to mono (a moving point source), resampled to 48 kHz and looped with an equal-power crossfade of
--xfade s when it is shorter than needed. Output: 48 kHz stereo 16-bit wav.
--offset pads silence in front (like tools/foley.py) so the file can go into cues.csv at start 0; without it the file
starts at shot time --start (default 0; negative = the sound leads the picture) and goes in at the shot's start in the cut.
--length is the rendered length after the offset (default: to the last keyframe).
"""
import argparse, csv
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

SR = 48000
C_SOUND = 343.0


def read_automation(path):
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    rows = []
    for r in csv.DictReader(lines):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if not r.get("time"):
            continue
        rows.append({"time": float(r["time"]), "pan": float(r.get("pan") or 0), "gain_db": float(r.get("gain_db") or 0),
                     "dist": float(r.get("dist") or 0), "rate": float(r["rate"]) if r.get("rate") else 1.0})
    if not rows:
        raise SystemExit(f"no rows in {path}")
    rows.sort(key=lambda z: z["time"])
    return rows


def curve(rows, key, t):
    return np.interp(t, [r["time"] for r in rows], [r[key] for r in rows])


def load_mono48(path):
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x


def loop_to(x, n, xf):
    """Repeat x until n samples, each repeat overlapping the previous by xf samples. The crossfade is equal-power,
    corrected for how alike the tail and the head are (correlation r: a^2 + b^2 + 2 r a b = 1), so neither noise-like
    nor tonal material dips or bumps at the join."""
    if len(x) >= n:
        return x[:n].copy()
    xf = int(min(xf, len(x) // 3))
    step = len(x) - xf
    out = np.zeros(n + len(x))
    fin = np.sin(np.linspace(0, np.pi / 2, xf)); fout = np.cos(np.linspace(0, np.pi / 2, xf))
    head, tail = x[:xf], x[len(x) - xf:]
    r = float(np.dot(head, tail) / (np.sqrt(np.dot(head, head) * np.dot(tail, tail)) + 1e-12))
    norm = np.sqrt(np.clip(fin ** 2 + fout ** 2 + 2 * r * fin * fout, 0.05, None))
    fin, fout = fin / norm, fout / norm
    pos, first = 0, True
    while pos < n:
        seg = x.copy()
        if not first:
            seg[:xf] *= fin
        seg[len(x) - xf:] *= fout
        out[pos:pos + len(x)] += seg
        pos += step; first = False
    return out[:n]


def cubic_read(src, pos):
    """Read src at fractional positions with 4-point cubic (Catmull-Rom) interpolation."""
    i = np.floor(pos).astype(np.int64); f = pos - i
    p = np.pad(src, (1, 3))
    xm1, x0, x1, x2 = p[i], p[i + 1], p[i + 2], p[i + 3]
    return x0 + 0.5 * f * (x1 - xm1 + f * (2 * xm1 - 5 * x0 + 4 * x1 - x2 + f * (3 * (x0 - x1) + x2 - xm1)))


def gauss_smooth(v, sigma_samples):
    if sigma_samples < 1:
        return v
    r = int(3 * sigma_samples)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2); k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("automation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=float, default=None, help="seconds rendered after --offset (default: to the last keyframe)")
    ap.add_argument("--offset", type=float, default=0.0, help="silence in front, e.g. the shot's start in the cut")
    ap.add_argument("--start", type=float, default=0.0, help="shot time at which the render begins (negative = pre-lap)")
    ap.add_argument("--far-db", type=float, default=-15.0, help="attenuation at dist 1")
    ap.add_argument("--near-hz", type=float, default=16000.0, help="low-pass at dist 0")
    ap.add_argument("--far-hz", type=float, default=2500.0, help="low-pass at dist 1")
    ap.add_argument("--bands", type=int, default=12, help="filtered copies crossfaded for the air absorption")
    ap.add_argument("--doppler", action="store_true")
    ap.add_argument("--near-m", type=float, default=10.0, help="metres at dist 0 (doppler only)")
    ap.add_argument("--far-m", type=float, default=250.0, help="metres at dist 1 (doppler only)")
    ap.add_argument("--doppler-smooth", type=float, default=0.25, help="seconds (gaussian sigma) of smoothing on the speed")
    ap.add_argument("--xfade", type=float, default=0.25, help="loop crossfade, seconds")
    ap.add_argument("--fade-in", type=float, default=0.01)
    ap.add_argument("--fade-out", type=float, default=0.01)
    a = ap.parse_args()

    rows = read_automation(a.automation)
    length = a.length if a.length is not None else max(rows[-1]["time"] - a.start, 0.5)
    n = int(round(length * SR))
    t = a.start + np.arange(n) / SR                         # shot time of every output sample

    pan = np.clip(curve(rows, "pan", t), -1, 1)
    dist = np.clip(curve(rows, "dist", t), 0, 1)
    gain_db = curve(rows, "gain_db", t) + a.far_db * dist
    rate = np.clip(curve(rows, "rate", t), 0.25, 4.0)

    if a.doppler:
        metres = a.near_m + dist * (a.far_m - a.near_m)
        v = np.gradient(gauss_smooth(metres, a.doppler_smooth * SR)) * SR     # m/s, + = receding
        v = np.clip(v, -0.5 * C_SOUND, 0.5 * C_SOUND)
        rate = rate * C_SOUND / (C_SOUND + v)

    # read the (looped) source at a varying rate: position = running sum of the rate
    pos = np.concatenate(([0.0], np.cumsum(rate[:-1])))
    src = loop_to(load_mono48(a.source), int(np.ceil(pos[-1])) + 8, int(a.xfade * SR))
    y = cubic_read(src, pos)

    # air absorption: crossfade between zero-phase low-passed copies (phase-aligned, so the blend has no comb)
    K = max(2, a.bands)
    cut = a.near_hz * (a.far_hz / a.near_hz) ** (np.arange(K) / (K - 1))
    u = dist * (K - 1)
    wet = np.zeros(n)
    for k in range(K):
        wk = np.clip(1 - np.abs(u - k), 0, None)
        if not wk.any():
            continue
        fc = min(cut[k], 0.45 * SR)
        wet += wk * sosfiltfilt(butter(1, fc, "low", fs=SR, output="sos"), y)
    y = wet * 10 ** (gain_db / 20)

    fi, fo = int(a.fade_in * SR), int(a.fade_out * SR)
    if fi > 1:
        y[:fi] *= np.linspace(0, 1, fi)
    if fo > 1:
        y[-fo:] *= np.linspace(1, 0, fo)

    th = (pan + 1) * np.pi / 4                               # equal-power pan
    st = np.stack([y * np.cos(th), y * np.sin(th)], axis=1)
    peak = float(np.abs(st).max())
    if peak > 0.98:
        st *= 0.98 / peak
        print(f"warning: peak {20 * np.log10(peak):+.1f} dBFS, scaled down to -0.2 dBFS")
    if a.offset > 0:
        st = np.concatenate([np.zeros((int(round(a.offset * SR)), 2)), st])
    sf.write(a.out, st, SR, subtype="PCM_16")
    semis = 12 * np.log2(rate)
    print(f"{a.out}: {length:.2f} s (+{a.offset:.2f} s offset), {len(rows)} keyframes, pan {pan.min():+.2f}..{pan.max():+.2f}, "
          f"dist {dist.min():.2f}..{dist.max():.2f}, low-pass {a.near_hz * (a.far_hz / a.near_hz) ** dist.max():.0f}.."
          f"{a.near_hz * (a.far_hz / a.near_hz) ** dist.min():.0f} Hz, pitch {semis.min():+.2f}..{semis.max():+.2f} st, "
          f"peak {20 * np.log10(max(peak, 1e-9)):+.1f} dBFS")


if __name__ == "__main__":
    main()
