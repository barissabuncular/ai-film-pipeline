"""Composite regions of an edited keyframe back onto the original (2026-10-12, for redo_redesign_v3 edits).
Qwen-Image-Edit returns 1392x752 for a 1280x720 input, with the frame cropped about 4% vertically (measured: y scale 1.04,
y offset -13 to -15 px). The edit is resized to the original size and then registered to it with an affine ECC fit.
Regions are ellipses or rectangles in original pixel coordinates, feathered with a Gaussian blur.
  python tools/rect_composite.py <orig.png> <edit.png> <out.png> --rect x0,y0,x1,y1 [--ellipse x0,y0,x1,y1] [--feather 31] [--check]
  --auto T   adds every region where the registered edit differs from the original by more than T (0-255, e.g. 30), dilated by
             --grow px, so only what the edit actually changed is taken (rects/ellipses can be combined with it)
--check prints the mean absolute difference outside the mask (alignment check: a well-aligned edit is < ~6) and writes <out>_mask.png.
"""
import argparse
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("orig"); ap.add_argument("edit"); ap.add_argument("out")
ap.add_argument("--rect", action="append", default=[]); ap.add_argument("--ellipse", action="append", default=[])
ap.add_argument("--feather", type=int, default=31); ap.add_argument("--check", action="store_true")
ap.add_argument("--auto", type=float, default=0); ap.add_argument("--grow", type=int, default=25)
a = ap.parse_args()
o = cv2.imread(a.orig); H, W = o.shape[:2]
e = cv2.resize(cv2.imread(a.edit), (W, H), interpolation=cv2.INTER_AREA)
g1, g2 = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (o, e))
_, M = cv2.findTransformECC(g1, g2, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
e = cv2.warpAffine(e, M, (W, H), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
m = np.zeros((H, W), np.float32)
for r in a.rect:
    x0, y0, x1, y1 = map(int, r.split(",")); m[y0:y1, x0:x1] = 1
for r in a.ellipse:
    x0, y0, x1, y1 = map(int, r.split(","))
    cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1.0, -1)
if a.auto:
    d = cv2.GaussianBlur(np.abs(o.astype(np.float32) - e).mean(2), (11, 11), 0)
    blob = (d > a.auto).astype(np.uint8)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blob)
    keep = np.zeros_like(blob)
    for i in range(1, n):
        if st[i, 4] >= 400:  # drop specks from global tone shifts
            keep[lab == i] = 1
    keep = cv2.dilate(keep, np.ones((2 * a.grow + 1, 2 * a.grow + 1), np.uint8))
    m = np.maximum(m, keep.astype(np.float32))
k = a.feather | 1; m = cv2.GaussianBlur(m, (k, k), 0)[..., None]
cv2.imwrite(a.out, (o * (1 - m) + e * m).round().astype(np.uint8))
if a.check:
    out_mask = m[..., 0] < 0.01
    print(f"mean abs diff outside mask: {np.abs(o.astype(np.float32) - e)[out_mask].mean():.2f}")
    cv2.imwrite(a.out[:-4] + "_mask.png", (m[..., 0] * 255).astype(np.uint8))
# Note (2026-10-12): for edits that change the whole frame (relight, regrade), do not composite: crop the edit to 16:9 around its
# centre and resize to 1280x720 (the edit then frames about 4% tighter than the original), as done for b35_09 and b18_04.
