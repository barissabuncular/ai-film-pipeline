"""Masked skin pass, done as a CPU composite (2026-10-08): the earlier full-image Z-Image pass (train/texture_pass.py) gave real skin but
drifted eye colour, age and expressions (Moreina's blind visual gate). Here only the face skin is taken from the passed image; eyes, brows,
nose tip, mouth, hair, kit and background stay pixel-identical to the original. Mask = face ellipse (YuNet box) minus ellipses around the
eyes+brows, nose tip and mouth, feathered. No face found -> the original is kept.
  python train/texture_composite.py <cand dir with <sub>__<name>__{orig,d15,d25}.png> <out dir>
Writes <out>/<sub>__<name>__{orig,c15,c25}.png and <out>/masks/.
"""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from idscore import Faces  # noqa: E402

src, out = sys.argv[1], sys.argv[2]; os.makedirs(f"{out}/masks", exist_ok=True)
fx = Faces(need_aura=False); n = nf = 0
for o in sorted(glob.glob(f"{src}/*__orig.png")):
    stem = os.path.basename(o)[:-len("__orig.png")]; orig = cv2.imread(o); H, W = orig.shape[:2]
    cv2.imwrite(f"{out}/{stem}__orig.png", orig); n += 1
    faces = fx.detect(orig)
    if not faces:
        for v in ("15", "25"):
            cv2.imwrite(f"{out}/{stem}__c{v}.png", orig)
        nf += 1; continue
    f = faces[0]; x, y, w, h = f[:4]; re_, le, nose, rm, lm = [(f[4 + 2 * i], f[5 + 2 * i]) for i in range(5)]
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(x + w / 2), int(y + h * 0.55)), (int(w * 0.46), int(h * 0.52)), 0, 0, 360, 1.0, -1)  # face skin area
    eye_r = (int(w * 0.20), int(h * 0.14))
    for ex, ey in (re_, le):  # eyes and brows
        cv2.ellipse(m, (int(ex), int(ey - h * 0.04)), eye_r, 0, 0, 360, 0.0, -1)
    cv2.ellipse(m, (int(nose[0]), int(nose[1])), (int(w * 0.11), int(h * 0.09)), 0, 0, 360, 0.0, -1)  # nose tip
    mx, my = (rm[0] + lm[0]) / 2, (rm[1] + lm[1]) / 2; mw = abs(lm[0] - rm[0]) / 2
    cv2.ellipse(m, (int(mx), int(my)), (int(mw * 1.35), int(h * 0.11)), 0, 0, 360, 0.0, -1)  # mouth
    k = max(3, int(w * 0.06) | 1); m = cv2.GaussianBlur(m, (k, k), 0)[..., None]
    cv2.imwrite(f"{out}/masks/{stem}.png", (m[..., 0] * 255).astype(np.uint8))
    for v in ("15", "25"):
        p = f"{src}/{stem}__d{v}.png"
        r = cv2.imread(p) if os.path.exists(p) else orig
        if r.shape != orig.shape:
            r = cv2.resize(r, (W, H), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(f"{out}/{stem}__c{v}.png", (orig * (1 - m) + r * m).astype(np.uint8))
print(f"{n} images, {nf} without a detected face (kept original)")
