"""Measure a tools/move.py render against its automation: does the sound sit, move and change the way the keyframes say?

Nobody has to listen to trust a moving-source stem. This reads the rendered wav back and reports, per window
(--win s long, one per video frame):
  pan    the rendered balance R - L (dB) turned back into an equal-power pan value (pan = 4/pi * atan(R/L) - 1),
         compared with the keyframed pan: Pearson r and the worst error in pan units. For a static pan r is undefined
         (no variance), so only the error is given.
  level  mono level vs the level the automation asks for (gain_db + far_db * dist, plus --fade shapes): r(dist, level)
         and the level change between the far and the near end, measured vs expected. The source's own wobble
         (rotor chop, a train's swell) is in the measured curve, so r < 1 is normal for a busy source.
  air    with --ref (the same render with the air-absorption low-pass parked, e.g. move.py --near-hz 21600 --far-hz 21600):
         2-8 kHz energy the distance low-pass removed, per window; r(dist, loss) and its value at the far and near end.
         The parked filter still costs up to 0.6 dB at 8 kHz; it is the same in every window, so differences are exact.
  chop   with --chop lo,hi and --source: the envelope modulation rate (rotor chop, flutter) per 1 s window of the render,
         against the rate predicted for that window: the source's own rate over exactly the source samples the window
         read (move.py's read position is rebuilt from the rate column and doppler), times samples read per output sample.
         Using the source's local rate keeps the recording's own wobble out of the comparison, so what is left is the
         doppler. --chop-band picks the envelope band (default 40-400 Hz, the rotor thump).
Stems made of several layers (renders of decorrelated copies of a source, e.g. a static vent plus a drifting plume) are
checked against the power sum of their layers: repeat --auto once per layer. Layers are assumed to have equal source
power (match their RMS when you cut them).

Usage:
  python tools/move_check.py sound/stems/<id>_move.wav --auto sound/stems/<id>_move.csv --start -0.5 --eval 0.4,4.6
         [--ref sound/stems/check/<id>_move_ref.wav] [--far-db -15] [--doppler --near-m 10 --far-m 250]
         [--chop 6,16 --source sound/lib/<source>.wav] [--fade in,-0.5,0.4 --fade out,4.6,5.0]
         [--track sound/stems/track/x.csv] [--png sound/stems/check/<id>_move.png] [--title "..."] [--json out.json]
--start is the shot time of the wav's first sample (negative for a pre-lap; 0 when the wav starts at the shot start).
--eval restricts the statistics to shot times t0..t1 (leave the dissolves out). Windows quieter than --floor dB under
the loudest window are skipped (a layer that is silent for part of the shot has no pan there).
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from move import read_automation, curve, gauss_smooth, C_SOUND  # noqa: E402

SR = 48000


def win_rms_db(x, centres, w):
    h = w // 2
    out = []
    for c in centres:
        seg = x[max(0, c - h):c + h]
        out.append(10 * np.log10(np.mean(seg ** 2) + 1e-20))
    return np.array(out)


def band(x, lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], "band", fs=SR, output="sos"), x)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def fade_curve(t, fades):
    g = np.ones_like(t)
    for kind, t0, t1 in fades:
        u = np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1)
        g *= np.sin(u * np.pi / 2) if kind == "in" else np.cos(u * np.pi / 2)
    return g


def chop_rate(env, fs, lo, hi):
    """Modulation rate of an envelope. The strongest line of the envelope's spectrum inside lo..hi (Hann, 16x zero-pad)
    picks the right period; the autocorrelation peak within +-12 % of it then refines it (parabolic interpolation).
    Autocorrelation alone can lock onto a neighbouring lag when one chop is irregular."""
    e = (env - env.mean()) * np.hanning(len(env))
    nfft = 16 * len(e)
    F = np.abs(np.fft.rfft(e, nfft)); f = np.fft.rfftfreq(nfft, 1 / fs)
    s = (f >= lo) & (f <= hi)
    if not s.any():
        return np.nan
    f0 = f[s][int(np.argmax(F[s]))]
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    a, b = max(2, int(fs / (f0 * 1.12))), int(fs / (f0 / 1.12)) + 1
    if b >= len(ac) - 1:
        return f0
    k = a + int(np.argmax(ac[a:b]))
    if k <= a or k >= b - 1:
        return f0
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-20)
    return fs / (k + d)


def envelope(x, lo=40, hi=400, fs_env=400):
    env = np.abs(hilbert(band(x, lo, hi)))
    d = SR // fs_env
    return env[:len(env) // d * d].reshape(-1, d).mean(axis=1), fs_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--auto", action="append", required=True, help="automation CSV (repeat once per layer)")
    ap.add_argument("--layer-db", default=None, help="comma list of extra dB per layer if the layers were mixed unequally")
    ap.add_argument("--start", type=float, default=0.0, help="shot time of the wav's first sample")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds of silence move.py --offset put in front")
    ap.add_argument("--eval", default=None, help="t0,t1 shot times used for the statistics")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--far-db", type=float, default=-15.0)
    ap.add_argument("--doppler", action="store_true")
    ap.add_argument("--near-m", type=float, default=10.0)
    ap.add_argument("--far-m", type=float, default=250.0)
    ap.add_argument("--doppler-smooth", type=float, default=0.25)
    ap.add_argument("--chop", default=None, help="lo,hi Hz range of the modulation rate to look for")
    ap.add_argument("--chop-band", default="40,400", help="Hz band whose envelope carries the chop (default: the rotor thump;"
                    " blade-slap harmonics higher up can lock onto other rates)")
    ap.add_argument("--source", default=None, help="the wav move.py read, for the predicted chop rate")
    ap.add_argument("--xfade", type=float, default=0.25, help="move.py --xfade (to rebuild its loop of the source)")
    ap.add_argument("--fade", action="append", default=[], help="in|out,t0,t1 (shot times) fades applied after move.py")
    ap.add_argument("--track", default=None, help="per-frame track CSV (tools/track.py) to draw and correlate")
    ap.add_argument("--win", type=float, default=0.25)
    ap.add_argument("--floor", type=float, default=30.0)
    ap.add_argument("--png", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    x, sr = sf.read(a.wav, always_2d=True)
    assert sr == SR, f"{a.wav}: {sr} Hz, expected {SR}"
    Lc, Rc = x[:, 0], x[:, 1]
    mono = x.mean(axis=1)
    t0_file = a.start - a.offset                      # shot time of sample 0
    hop = SR // 32
    w = int(a.win * SR)
    centres = np.arange(w // 2, len(mono) - w // 2, hop)
    tc = t0_file + centres / SR

    layers = [read_automation(p) for p in a.auto]
    ldb = [float(v) for v in a.layer_db.split(",")] if a.layer_db else [0.0] * len(layers)
    fades = [(f.split(",")[0], float(f.split(",")[1]), float(f.split(",")[2])) for f in a.fade]

    # expected per layer at every sample, then power-summed per window
    ts = t0_file + np.arange(len(mono)) / SR
    fg = fade_curve(ts, fades) if fades else np.ones_like(ts)
    PL = np.zeros(len(mono)); PR = np.zeros(len(mono))
    for rows, extra in zip(layers, ldb):
        pan = np.clip(curve(rows, "pan", ts), -1, 1)
        dist = np.clip(curve(rows, "dist", ts), 0, 1)
        g = 10 ** ((curve(rows, "gain_db", ts) + a.far_db * dist + extra) / 20) * fg
        th = (pan + 1) * np.pi / 4
        PL += (g * np.cos(th)) ** 2; PR += (g * np.sin(th)) ** 2
    exp_L = win_rms_db(np.sqrt(PL), centres, w); exp_R = win_rms_db(np.sqrt(PR), centres, w)
    exp_bal = exp_R - exp_L
    exp_pan = 4 / np.pi * np.arctan(10 ** (exp_bal / 20)) - 1
    exp_lev = 10 * np.log10((10 ** (exp_L / 10) + 10 ** (exp_R / 10)) / 2)

    mL = win_rms_db(Lc, centres, w); mR = win_rms_db(Rc, centres, w)
    m_bal = mR - mL
    m_pan = 4 / np.pi * np.arctan(10 ** (m_bal / 20)) - 1
    m_lev = 10 * np.log10((10 ** (mL / 10) + 10 ** (mR / 10)) / 2)

    lo_t, hi_t = (float(v) for v in a.eval.split(",")) if a.eval else (tc[0], tc[-1])
    sel = (tc >= lo_t) & (tc <= hi_t) & (m_lev > m_lev.max() - a.floor)
    main_rows = layers[-1]
    kf_pan = np.clip(curve(main_rows, "pan", tc), -1, 1)
    kf_dist = np.clip(curve(main_rows, "dist", tc), 0, 1)

    res = {"wav": a.wav, "eval": [lo_t, hi_t], "windows": int(sel.sum())}
    static = np.ptp(exp_pan[sel]) < 0.01                   # a static pan has nothing to correlate with
    res["pan_static"] = bool(static)
    res["pan_r_expected"] = None if static else pearson(exp_pan[sel], m_pan[sel])
    res["pan_r_keyframes_last_layer"] = None if static else pearson(kf_pan[sel], m_pan[sel])
    res["pan_max_err"] = float(np.max(np.abs(m_pan[sel] - exp_pan[sel])))
    res["pan_measured_range"] = [float(m_pan[sel].min()), float(m_pan[sel].max())]
    res["pan_expected_range"] = [float(exp_pan[sel].min()), float(exp_pan[sel].max())]
    res["balance_db_range"] = [float(m_bal[sel].min()), float(m_bal[sel].max())]
    res["level_r_dist"] = pearson(kf_dist[sel], m_lev[sel])
    res["level_r_expected"] = pearson(exp_lev[sel], m_lev[sel])
    # far vs near end: mean over the windows in the top / bottom 20 % of dist inside the evaluation span
    d = kf_dist[sel]
    if np.ptp(d) > 1e-6:
        far = d >= d.max() - 0.2 * np.ptp(d); near = d <= d.min() + 0.2 * np.ptp(d)
        res["dist_far_near"] = [float(d[far].mean()), float(d[near].mean())]
        res["level_change_far_to_near_db"] = float(m_lev[sel][near].mean() - m_lev[sel][far].mean())
        res["level_change_expected_db"] = float(exp_lev[sel][near].mean() - exp_lev[sel][far].mean())
    else:
        far = near = None

    loss = None
    if a.ref:
        r, rs = sf.read(a.ref, always_2d=True)
        assert rs == SR and len(r) == len(x), "ref must be the same render with the filter parked"
        b_out = win_rms_db(band(mono, 2000, 8000), centres, w)
        b_ref = win_rms_db(band(r.mean(axis=1), 2000, 8000), centres, w)
        loss = b_out - b_ref
        tilt = b_out - m_lev                               # 2-8 kHz share of the render, dB
        res["air_r_dist"] = pearson(d, loss[sel])
        res["air_loss_range_db"] = [float(loss[sel].min()), float(loss[sel].max())]
        if far is not None:
            res["air_loss_far_near_db"] = [float(loss[sel][far].mean()), float(loss[sel][near].mean())]
            res["band_2_8k_change_far_to_near_db"] = float(b_out[sel][near].mean() - b_out[sel][far].mean())
            res["tilt_2_8k_change_far_to_near_db"] = float(tilt[sel][near].mean() - tilt[sel][far].mean())

    chop = None
    if a.chop:
        clo, chi = (float(v) for v in a.chop.split(","))
        blo, bhi = (float(v) for v in a.chop_band.split(","))
        env, fe = envelope(mono, blo, bhi)
        cw = fe; hopc = fe // 8                              # 1 s windows, 1/8 s hop
        ct, cm = [], []
        for s in range(0, len(env) - cw, hopc):
            ct.append(t0_file + (s + cw / 2) / fe); cm.append(chop_rate(env[s:s + cw], fe, clo, chi))
        ct, cm = np.array(ct), np.array(cm)
        # playback rate exactly as move.py computes it (rate column x doppler), from the first rendered sample on
        i_r = int(round(a.offset * SR))
        tr = ts[i_r:]
        rows = layers[-1]
        rate = np.clip(curve(rows, "rate", tr), 0.25, 4.0)
        if a.doppler:
            dist = np.clip(curve(rows, "dist", tr), 0, 1)
            metres = a.near_m + dist * (a.far_m - a.near_m)
            v = np.gradient(gauss_smooth(metres, a.doppler_smooth * SR)) * SR
            rate = rate * C_SOUND / (C_SOUND + np.clip(v, -0.5 * C_SOUND, 0.5 * C_SOUND))
        pos = np.concatenate(([0.0], np.cumsum(rate[:-1])))   # source sample read at every output sample
        pred_w = np.full(len(ct), np.nan); nominal = None
        if a.source:
            from move import load_mono48, loop_to
            src = loop_to(load_mono48(a.source), int(np.ceil(pos[-1])) + 8, int(a.xfade * SR))
            senv, _ = envelope(src, blo, bhi)
            nominal = float(np.nanmedian([chop_rate(senv[s:s + cw], fe, clo, chi) for s in range(0, len(senv) - cw, cw // 2)]))
            # predicted = the source's own rate over exactly the samples this window read, x the samples read per output sample
            for j, c in enumerate(ct):
                i0 = int((c - 0.5 - t0_file) * SR) - i_r; i1 = int((c + 0.5 - t0_file) * SR) - i_r
                if i0 < 0 or i1 >= len(pos):
                    continue
                p0, p1 = pos[i0], pos[i1]
                r_src = chop_rate(senv[int(p0 * fe / SR):int(p1 * fe / SR)], fe, clo, chi)
                pred_w[j] = r_src * (p1 - p0) / (i1 - i0)
        cs = (ct >= lo_t) & (ct <= hi_t) & np.isfinite(cm) & np.isfinite(pred_w)
        chop = (ct, cm, pred_w)
        res["chop_band_hz"] = [blo, bhi]
        res["chop_source_median_hz"] = nominal
        res["chop_r_predicted"] = pearson(pred_w[cs], cm[cs])
        res["chop_measured_range_hz"] = [float(np.nanmin(cm[cs])), float(np.nanmax(cm[cs]))]
        res["chop_predicted_range_hz"] = [float(np.nanmin(pred_w[cs])), float(np.nanmax(pred_w[cs]))]
        res["chop_mean_abs_err_hz"] = float(np.nanmean(np.abs(cm[cs] - pred_w[cs])))
        res["chop_max_abs_err_hz"] = float(np.nanmax(np.abs(cm[cs] - pred_w[cs])))
        semis = 12 * np.log2(rate)
        res["doppler_semitones_range"] = [float(semis.min()), float(semis.max())]

    trk = None
    if a.track:
        import csv as _csv
        with open(a.track, encoding="utf-8") as f:
            rr = list(_csv.DictReader(f))
        tt = np.array([float(r["time"]) for r in rr]); tx = np.array([float(r["x"]) for r in rr])
        tdist = np.array([float(r["dist"]) for r in rr])
        trk = (tt, tx, tdist)
        res["pan_r_track"] = None if static else pearson(np.interp(tc[sel], tt, tx), m_pan[sel])

    txt = json.dumps(res, indent=1)
    print(txt)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(txt)

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 4 + (loss is not None) + (chop is not None)
        fig, ax = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=True)
        i = 0
        def shade(axx):
            if a.eval:
                axx.axvspan(tc[0], lo_t, color="0.9", zorder=0); axx.axvspan(hi_t, tc[-1], color="0.9", zorder=0)
        kt = [r["time"] for r in main_rows]
        ax[i].plot(tc, kf_pan, color="#e8643c", lw=2, label="keyframed pan (last layer)")
        ax[i].plot(kt, [r["pan"] for r in main_rows], "o", color="#e8643c", ms=4)
        if len(layers) > 1:
            ax[i].plot(tc, exp_pan, color="0.4", lw=1.5, ls="--", label="expected pan of the layer sum")
        ax[i].plot(tc, np.where(m_lev > m_lev.max() - a.floor, m_pan, np.nan), color="#2a78d6", lw=1.2, label="measured pan (from R-L)")
        if trk:
            ax[i].plot(trk[0], trk[1], color="#1a9c5b", lw=1, label="tracked x")
        ax[i].set_ylim(-1, 1); ax[i].axhline(0, color="0.6", lw=0.5); ax[i].set_ylabel("pan / x"); ax[i].legend(fontsize=7, loc="best")
        rp = res["pan_r_expected"]
        ax[i].set_title(f"pan: r(expected, measured) = {rp:.3f}" if rp is not None else
                        f"pan (static): max error {res['pan_max_err']:.3f}", fontsize=9, loc="left"); shade(ax[i]); i += 1
        ax[i].plot(tc, kf_dist, color="#e8643c", lw=2, label="dist keyframes"); ax[i].plot(kt, [r["dist"] for r in main_rows], "o", color="#e8643c", ms=4)
        ax[i].set_ylim(0, 1); ax[i].set_ylabel("dist (0 near)"); ax[i].legend(fontsize=7); shade(ax[i]); i += 1
        ax[i].plot(tc, mL, color="#2a78d6", lw=1, label="L"); ax[i].plot(tc, mR, color="#e8643c", lw=1, label="R")
        off = np.median(m_lev[sel] - exp_lev[sel])
        ax[i].plot(tc, exp_lev + off, color="0.3", lw=1.5, ls="--", label="expected level (shifted)")
        rl = res["level_r_expected"]
        ax[i].set_title(f"level ({a.win:.2f} s RMS)" + (f": r(expected, measured) = {rl:.3f}" if rl is not None else ""), fontsize=9, loc="left")
        ax[i].set_ylabel("dBFS"); ax[i].legend(fontsize=7); ax[i].set_ylim(max(m_lev.max() - 45, -90), m_lev.max() + 3); shade(ax[i]); i += 1
        ax[i].plot(tc, exp_bal, color="#e8643c", lw=2, label="expected R-L"); ax[i].plot(tc, np.where(m_lev > m_lev.max() - a.floor, m_bal, np.nan), color="#2a78d6", lw=1, ls="--", label="measured R-L")
        ax[i].set_ylabel("R-L dB"); ax[i].axhline(0, color="0.6", lw=0.5); ax[i].legend(fontsize=7); shade(ax[i]); i += 1
        if loss is not None:
            ax[i].plot(tc, loss, color="#2a78d6", lw=1.5, label="2-8 kHz lost to the distance low-pass")
            ax[i].set_ylabel("dB"); ax[i].legend(fontsize=7)
            ra = res["air_r_dist"]
            ax[i].set_title(f"air absorption" + (f": r(dist, loss) = {ra:.3f}" if ra is not None else ""), fontsize=9, loc="left"); shade(ax[i]); i += 1
        if chop is not None:
            ax[i].plot(chop[0], chop[1], color="#2a78d6", lw=1.5, label="measured modulation rate (1 s windows)")
            ax[i].plot(chop[0], chop[2], color="#e8643c", lw=1.5, ls="--", label="source rate x doppler playback rate")
            ax[i].set_ylabel("Hz"); ax[i].legend(fontsize=7)
            rc = res["chop_r_predicted"]
            ax[i].set_title("rotor chop rate (doppler check)" + (f": r = {rc:.3f}" if rc is not None else ""), fontsize=9, loc="left"); shade(ax[i]); i += 1
        ax[-1].set_xlabel("shot time (s); grey = outside the evaluation span (dissolves / fades)")
        fig.suptitle(a.title or os.path.basename(a.wav), fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(a.png, dpi=90)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
