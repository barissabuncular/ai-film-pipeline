"""Track where a sound source is on screen, frame by frame, and reduce it to movement-automation keyframes for tools/move.py.

Modes (--mode):
  bright  brightest blob: searchlights, headlights, lamps, muzzle flashes. Position = centre of the hottest core connected to
          the peak (--core), so it stays on the lamp while its beam swings; size = diameter of the glow around it (--thresh,
          median radius over 48 directions, so a beam leaving the lamp does not count).
  motion  centre of frame-to-frame change inside --region (a car, a person, anything that moves against a still background).
          The median change is subtracted first, so camera shake and the Wan 4-frame decode jumps count less.
  manual  no tracking: interpolate typed keyframes (--manual). Use it when the source is off screen or the tracker fails.
In bright/motion mode, frames whose confidence is under --min-conf are gaps; they are filled from --manual keyframes when given,
otherwise interpolated from the neighbours. --prefer-manual makes typed keyframes win wherever they cover the time.

Per-frame CSV (--out) columns:  frame,time,x,y,size,dist,conf,src
  frame  0-based index, same numbering as the tools/spot.py sheets; time = frame / fps from the shot start
  x      -1 left edge .. 0 centre .. +1 right edge (beyond +-1 = off screen, typed keyframes only)
  y      -1 top .. +1 bottom
  size   bright: glow diameter / frame width; motion: spread of the moving area / frame width. Bigger usually means nearer.
  dist   0 near .. 1 far, from size (see below) or from the manual dist column
  conf   0..1 tracker confidence;  src = track | manual | interp
Manual keyframe CSV (--manual) columns: time (or frame),x[,y][,size][,dist][,note]   '#' lines are comments.
Automation CSV (--keyframes N --auto <csv>) columns: time,pan,gain_db,dist,x,y,size,note  -> feed straight to tools/move.py
  pan  = x * --pan-scale, clipped to -1..1
  dist = the manual dist column if given, else from size: largest size -> first value of --dist-range, smallest ->
         second, in between by --dist-law (inverse: distance ~ 1/size, the default; log). 0 near .. 1 far, not metres.
  Keyframes are picked greedily where straight-line interpolation of pan/dist would be most wrong, so N points follow
  the curve as closely as they can. Check them against the frames and edit by hand; the tracker is a first draft.

Usage:
  python tools/track.py shots/<id>/final_1080p.mp4 --out review/track_<id>/track.csv --mode bright --region 0.1,0,0.7,0.6
         [--smooth 3] [--thresh 0.85] [--core 0.985] [--min-conf 0.2] [--work-width 480] [--keyframes 8 --auto sound/move_x_auto.csv]
         [--manual kf.csv [--prefer-manual]] [--pan-scale 1.0] [--dist-range 0.2,0.8 --dist-law inverse] [--sheet review/track_<id>/track_sheet.png]
--sheet draws the tracked point on a contact sheet of every --sheet-step-th frame, so the track can be checked by eye.
"""
import argparse, csv, os
import cv2
import numpy as np


def read_frames(path, work_w):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if work_w and w > work_w:
            fr = cv2.resize(fr, (work_w, int(round(h * work_w / w))), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    return frames, fps


def count_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, fr = cap.read()
    cap.release()
    return n, fps, (fr.shape[1], fr.shape[0]) if ok else (1920, 1080)


def region_px(region, w, h):
    x0, y0, x1, y1 = region
    return int(x0 * w), int(y0 * h), max(int(x0 * w) + 2, int(x1 * w)), max(int(y0 * h) + 2, int(y1 * h))


def track_bright(frames, region, thresh, blur, core):
    """Centre = brightness-weighted centroid of the hottest core (pixels within the top (1-core) of the peak-to-median
    range, connected to the peak). The lamp itself is hotter than the beam or streak leaving it, so the point stays on
    the lamp while the beam swings. Size = 2 x the median radius, over 48 directions from the centre, at which brightness
    falls under median + thresh x (peak - median): the glow grows as the source nears, and the beam covers only a few
    directions, so it does not inflate the size."""
    out = []
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for fr in frames:
        h, w = fr.shape[:2]
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), max(0.5, blur * w))
        X0, Y0, X1, Y1 = region_px(region, w, h)
        sub = g[Y0:Y1, X0:X1]
        _, peak, _, (px, py) = cv2.minMaxLoc(sub)
        base = float(np.median(sub))
        hot = sub >= base + (peak - base) * core
        _, labels = cv2.connectedComponents(hot.astype(np.uint8), connectivity=8)
        comp = labels == labels[py, px]
        ys, xs = np.nonzero(comp)
        wts = sub[comp] - (base + (peak - base) * core) + 1e-3
        cx = float((xs * wts).sum() / wts.sum()); cy = float((ys * wts).sum() / wts.sum())
        thr = base + (peak - base) * thresh
        R = int(0.3 * w)
        r = np.arange(R, dtype=np.float32)
        xx = (cx + np.outer(np.cos(ang), r)).round().astype(int)
        yy = (cy + np.outer(np.sin(ang), r)).round().astype(int)
        outside = (xx < 0) | (yy < 0) | (xx >= sub.shape[1]) | (yy >= sub.shape[0])
        below = outside | (sub[np.clip(yy, 0, sub.shape[0] - 1), np.clip(xx, 0, sub.shape[1] - 1)] < thr)
        radii = np.where(below.any(1), below.argmax(1), R).astype(float)
        rad = float(np.median(radii))
        conf = float(np.clip((peak - base) / max(1.0, 255.0 - base), 0, 1))
        out.append(((cx + X0) / w * 2 - 1, (cy + Y0) / h * 2 - 1, 2 * rad / w, conf))
    return out


def track_motion(frames, region, thresh, blur):
    out, prev = [], None
    for fr in frames:
        h, w = fr.shape[:2]
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), max(0.5, blur * w))
        X0, Y0, X1, Y1 = region_px(region, w, h)
        sub = g[Y0:Y1, X0:X1]
        if prev is None:
            prev = sub
            out.append(None)
            continue
        d = np.abs(sub - prev)
        prev = sub
        d = np.clip(d - np.median(d), 0, None)          # remove global flicker / decode jumps
        cut = np.quantile(d, thresh) if 0 < thresh < 1 else d.max() * 0.3
        wts = np.where(d >= cut, d, 0.0) ** 2
        s = wts.sum()
        if s <= 0:
            out.append((0.0, 0.0, 0.0, 0.0))
            continue
        ys, xs = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
        cx = float((xs * wts).sum() / s); cy = float((ys * wts).sum() / s)
        spread = np.sqrt(float((((xs - cx) ** 2 + (ys - cy) ** 2) * wts).sum() / s))
        conf = float(np.clip(np.quantile(d, 0.995) / 32.0, 0, 1))  # ~32 grey levels of change = sure
        out.append(((cx + X0) / w * 2 - 1, (cy + Y0) / h * 2 - 1, 2 * spread / w, conf))
    if len(out) > 1 and out[0] is None:
        out[0] = out[1]
    return [o if o is not None else (0.0, 0.0, 0.0, 0.0) for o in out]


def read_manual(path, fps):
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    for r in csv.DictReader(lines):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("time"):
            t = float(r["time"])
        elif r.get("frame"):
            t = float(r["frame"]) / fps
        else:
            continue
        rows.append((t, {k: float(r[k]) for k in ("x", "y", "size", "dist") if r.get(k)}))
    rows.sort(key=lambda z: z[0])
    return rows


def manual_curve(rows, times, key):
    pts = [(t, v[key]) for t, v in rows if key in v]
    if not pts:
        return None, None
    tt = np.array([p[0] for p in pts]); vv = np.array([p[1] for p in pts])
    return np.interp(times, tt, vv), (tt[0], tt[-1])


def fill_gaps(v):
    v = v.copy()
    good = ~np.isnan(v)
    if not good.any():
        return np.zeros_like(v)
    idx = np.arange(len(v))
    v[~good] = np.interp(idx[~good], idx[good], v[good])
    return v


def gauss_smooth(v, sigma):
    if sigma <= 0 or len(v) < 3:
        return v
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.pad(v, r, mode="edge")
    return np.convolve(p, k, mode="valid")


def reduce_keyframes(times, curves, n):
    """Greedy: start with both ends, keep adding the frame where linear interpolation is furthest off (max over curves)."""
    keep = [0, len(times) - 1]
    while len(keep) < min(n, len(times)):
        ks = sorted(keep)
        err = np.zeros(len(times))
        for c in curves:
            err = np.maximum(err, np.abs(c - np.interp(times, times[ks], c[ks])))
        err[ks] = -1
        i = int(err.argmax())
        if err[i] <= 1e-4:
            break
        keep.append(i)
    return sorted(keep)


def draw_sheet(clip, xs, ys, srcs, fps, path, step, thumb=240, cols=8):
    frames, _ = read_frames(clip, thumb)
    idx = list(range(0, len(frames), step))
    th = frames[0].shape[0]
    rows = -(-len(idx) // cols)
    sheet = np.full((rows * (th + 16), cols * thumb, 3), 20, np.uint8)
    col = {"track": (0, 255, 255), "manual": (255, 0, 255), "interp": (0, 128, 255)}
    for j, i in enumerate(idx):
        im = frames[i].copy()
        px = int((xs[i] + 1) / 2 * thumb); py = int((ys[i] + 1) / 2 * th)
        px = min(max(px, 3), thumb - 4); py = min(max(py, 3), th - 4)
        c = col.get(srcs[i], (255, 255, 255))
        cv2.drawMarker(im, (px, py), c, cv2.MARKER_CROSS, 18, 2)
        cv2.circle(im, (px, py), 9, c, 1)
        x = (j % cols) * thumb; y = (j // cols) * (th + 16)
        sheet[y + 16:y + 16 + th, x:x + thumb] = im
        cv2.putText(sheet, f"{i} {i / fps:.2f}s x{xs[i]:+.2f}", (x + 3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
    cv2.imwrite(path, sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", required=True, help="per-frame CSV")
    ap.add_argument("--mode", choices=["bright", "motion", "manual"], default="bright")
    ap.add_argument("--region", default="0,0,1,1", help="x0,y0,x1,y1 as fractions of width/height")
    ap.add_argument("--thresh", type=float, default=None, help="bright: glow edge for the size, as a fraction of the "
                    "median-to-peak range (0.85); motion: quantile of change kept (0.98)")
    ap.add_argument("--core", type=float, default=0.985, help="bright: pixels above this fraction of the median-to-peak "
                    "range, connected to the peak, form the lamp core whose centre is the position")
    ap.add_argument("--blur", type=float, default=0.01, help="blur sigma as a fraction of the working width")
    ap.add_argument("--min-conf", type=float, default=0.2)
    ap.add_argument("--smooth", type=float, default=3.0, help="gaussian sigma in frames (0 = off)")
    ap.add_argument("--work-width", type=int, default=480)
    ap.add_argument("--manual", default=None, help="typed keyframes: time|frame,x[,y][,size][,dist]")
    ap.add_argument("--prefer-manual", action="store_true")
    ap.add_argument("--keyframes", type=int, default=0)
    ap.add_argument("--auto", default=None, help="automation CSV for tools/move.py (needs --keyframes)")
    ap.add_argument("--pan-scale", type=float, default=1.0)
    ap.add_argument("--dist-range", default="0.2,0.8", help="dist at the largest,smallest size")
    ap.add_argument("--dist-law", choices=["inverse", "log"], default="inverse",
                    help="inverse: distance ~ 1/size (physical); log: dist linear in log(size)")
    ap.add_argument("--sheet", default=None, help="contact sheet with the tracked point drawn on")
    ap.add_argument("--sheet-step", type=int, default=4)
    a = ap.parse_args()
    region = [float(v) for v in a.region.split(",")]

    if a.mode == "manual":
        if not a.manual:
            raise SystemExit("--mode manual needs --manual <csv>")
        n, fps, _ = count_frames(a.clip)
        raw = [(np.nan, np.nan, np.nan, 0.0)] * n
    else:
        frames, fps = read_frames(a.clip, a.work_width)
        n = len(frames)
        if a.mode == "bright":
            raw = track_bright(frames, region, a.thresh if a.thresh is not None else 0.85, a.blur, a.core)
        else:
            raw = track_motion(frames, region, a.thresh if a.thresh is not None else 0.98, a.blur)
        del frames
    times = np.arange(n) / fps
    conf = np.array([r[3] for r in raw], float)
    bad = conf < a.min_conf
    x = np.where(bad, np.nan, [r[0] for r in raw]).astype(float)
    y = np.where(bad, np.nan, [r[1] for r in raw]).astype(float)
    sz = np.where(bad, np.nan, [r[2] for r in raw]).astype(float)
    src = np.where(bad, "interp", "track").astype(object)

    mdist = None
    if a.manual:
        rows = read_manual(a.manual, fps)
        for key, arr in (("x", x), ("y", y), ("size", sz)):
            curve, span = manual_curve(rows, times, key)
            if curve is None:
                continue
            inside = (times >= span[0] - 1e-9) & (times <= span[1] + 1e-9)
            use = (np.isnan(arr) | (inside & a.prefer_manual)) if a.mode != "manual" else np.ones(n, bool)
            arr[use] = curve[use]
            if key == "x":
                src[use] = "manual"
        mdist, _ = manual_curve(rows, times, "dist")

    x, y = fill_gaps(x), fill_gaps(y)
    sz = fill_gaps(sz) if not np.all(np.isnan(sz)) else np.full(n, 0.05)
    if a.mode != "manual":
        x, y = gauss_smooth(x, a.smooth), gauss_smooth(y, a.smooth)
        sz = np.exp(gauss_smooth(np.log(np.clip(sz, 1e-4, None)), a.smooth))

    pan = np.clip(x * a.pan_scale, -1, 1)
    if mdist is not None:
        dist = np.clip(mdist, 0, 1)
    else:
        d_near, d_far = [float(v) for v in a.dist_range.split(",")]
        lo, hi = sz.min(), sz.max()
        if hi / max(lo, 1e-6) < 1.05:
            dist = np.full(n, (d_near + d_far) / 2)
        elif a.dist_law == "inverse":      # apparent size ~ 1/distance, and tools/move.py maps dist linearly to metres
            dist = d_near + (d_far - d_near) * (1 / sz - 1 / hi) / (1 / lo - 1 / hi)
        else:
            dist = d_near + (d_far - d_near) * (np.log(hi) - np.log(sz)) / (np.log(hi) - np.log(lo))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time", "x", "y", "size", "dist", "conf", "src"])
        for i in range(n):
            w.writerow([i, f"{times[i]:.4f}", f"{x[i]:.4f}", f"{y[i]:.4f}", f"{sz[i]:.5f}", f"{dist[i]:.4f}", f"{conf[i]:.3f}", src[i]])
    counts = {k: int((src == k).sum()) for k in ("track", "manual", "interp")}
    print(f"{a.out}: {n} frames at {fps:g} fps, mode {a.mode}, frames tracked/manual/interpolated "
          f"{counts['track']}/{counts['manual']}/{counts['interp']}"
          f"  x {x.min():+.2f}..{x.max():+.2f}  size {sz.min():.3f}..{sz.max():.3f}  dist {dist.min():.2f}..{dist.max():.2f}")

    if a.keyframes and a.auto:
        keep = reduce_keyframes(times, [pan, dist], a.keyframes)
        with open(a.auto, "w", newline="") as fh:
            fh.write(f"# automation from tools/track.py {os.path.basename(a.clip)} mode={a.mode} region={a.region}\n")
            w = csv.writer(fh)
            w.writerow(["time", "pan", "gain_db", "dist", "x", "y", "size", "note"])
            for i in keep:
                w.writerow([f"{times[i]:.3f}", f"{pan[i]:+.3f}", "0", f"{dist[i]:.3f}", f"{x[i]:+.3f}", f"{y[i]:+.3f}",
                            f"{sz[i]:.4f}", f"frame {i} ({src[i]})"])
        print(f"{a.auto}: {len(keep)} keyframes at frames {keep}")

    if a.sheet:
        draw_sheet(a.clip, x, y, list(src), fps, a.sheet, a.sheet_step)
        print(f"{a.sheet}")


if __name__ == "__main__":
    main()
