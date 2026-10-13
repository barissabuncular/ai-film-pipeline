"""Face-identity scores for character LoRA work (CPU only). See story/diablo/episodes/ep01_the_butcher/lora_plan.md.

Detection: OpenCV YuNet on a copy scaled so the biggest face is about 250 px (YuNet works on 10-300 px faces); the
landmarks are scaled back and FaceRecognizerSF.alignCrop cuts a 112x112 aligned face from the full-resolution image.
Scores: AuraFace glintr100 (Apache 2.0, main) and OpenCV SFace (second opinion), cosine similarity to an anchor
centroid. The InsightFace model zoo is never used (non-commercial weights).

Modes:
  sheet     numbered contact sheet of face crops for the user to label (same person: yes / no)
            idscore.py sheet --anchor A.png --images <globs> --out review/label_aidan --size 160
  score     score images against the anchor(s): writes a CSV (file, face_px, aura, sface)
            idscore.py score --anchor A.png [B.png ...] --images <globs> --out scores.csv [--pick left]
  validate  AUC and best threshold of each scorer against the user's labels
            idscore.py validate --scores scores.csv --labels labels.csv   (labels: file,label with yes/no)

Run with the ComfyUI Python. The GPU is never used.
"""
import argparse, csv, glob, math, os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
YUNET = os.path.join(MODELS, "opencv", "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODELS, "opencv", "face_recognition_sface_2021dec.onnx")
AURA = os.path.join(MODELS, "auraface", "glintr100.onnx")
AURA_SHA = "a7933ea53301"


def expand(patterns):
    out = []
    for p in patterns:
        out += sorted(glob.glob(p, recursive=True)) if any(c in p for c in "*?[") else [p]
    return [f for f in out if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]


class Faces:
    def __init__(self, need_aura=True):
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.6, 0.3, 5000)
        self.sf = cv2.FaceRecognizerSF.create(SFACE, "")
        self.aura = None
        if need_aura:
            import hashlib
            h = hashlib.sha256(open(AURA, "rb").read()).hexdigest()
            if not h.startswith(AURA_SHA):
                sys.exit(f"AuraFace hash {h[:12]} != pinned {AURA_SHA}; refusing to run")
            self.aura = cv2.dnn.readNetFromONNX(AURA)
            self.aura.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.aura.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, img):
        """All faces as rows [x, y, w, h, 10 landmark coords, score] in full-resolution coordinates, biggest first."""
        H, W = img.shape[:2]
        # first pass at a moderate size to learn the face scale, then rescale so the biggest face is ~250 px
        s1 = min(1.0, 1024 / max(H, W))
        small = cv2.resize(img, (int(W * s1), int(H * s1))) if s1 < 1 else img
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, f1 = self.det.detect(small)
        if f1 is None or len(f1) == 0:
            return []
        biggest = max(f1[:, 2].max(), f1[:, 3].max()) / s1
        s2 = min(1.0, 250 / biggest) if biggest > 0 else s1
        if abs(s2 - s1) > 0.05:
            small = cv2.resize(img, (max(1, int(W * s2)), max(1, int(H * s2))))
            self.det.setInputSize((small.shape[1], small.shape[0]))
            _, f2 = self.det.detect(small)
            if f2 is not None and len(f2):
                f1, s1 = f2, s2
        faces = f1.copy()
        faces[:, :14] /= s1
        return sorted(faces.tolist(), key=lambda f: -f[2] * f[3])

    def embed(self, img, face):
        crop = self.sf.alignCrop(img, np.array(face, dtype=np.float32))
        v_sf = self.sf.feature(crop).flatten()
        v_sf /= np.linalg.norm(v_sf) + 1e-9
        v_au = None
        if self.aura is not None:
            blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
            self.aura.setInput(blob)
            v_au = self.aura.forward().flatten()
            v_au /= np.linalg.norm(v_au) + 1e-9
        return crop, v_au, v_sf


def centroid(vs):
    c = np.mean(np.stack(vs), 0)
    return c / (np.linalg.norm(c) + 1e-9)


def anchor_vectors(fx, anchors):
    au, sf = [], []
    for a in anchors:
        img = cv2.imread(a)
        faces = fx.detect(img)
        if not faces:
            sys.exit(f"no face found in anchor {a}")
        _, va, vs = fx.embed(img, faces[0])
        au.append(va); sf.append(vs)
        print(f"anchor {os.path.basename(a)}: face {int(faces[0][2])}x{int(faces[0][3])} px")
    return centroid(au) if au[0] is not None else None, centroid(sf)


def cmd_score(a):
    fx = Faces()
    c_au, c_sf = anchor_vectors(fx, a.anchor)
    rows = []
    for f in expand(a.images):
        img = cv2.imread(f)
        if img is None:
            continue
        faces = fx.detect(img)
        if not faces:
            rows.append({"file": f, "face_px": 0, "aura": "", "sface": ""}); continue
        face = faces[0]
        if a.pick == "left":  # two-shots with the hero on the left: left-most confident face of usable size
            face = min([fc for fc in faces if fc[14] >= 0.8 and max(fc[2], fc[3]) >= 40] or faces, key=lambda fc: fc[0] + fc[2] / 2)
        _, va, vs = fx.embed(img, face)
        rows.append({"file": f, "face_px": int(max(face[2], face[3])),
                     "aura": round(float(va @ c_au), 4), "sface": round(float(vs @ c_sf), 4)})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "face_px", "aura", "sface"]); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} images scored -> {a.out}")


def cmd_sheet(a):
    from PIL import Image, ImageDraw, ImageFont
    fx = Faces(need_aura=False)
    files = expand(a.images)
    os.makedirs(a.out, exist_ok=True)
    S = a.size
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(14, S // 8))
    tiles, index = [], []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        faces = fx.detect(img)
        if not faces:
            continue
        good = [fc for fc in faces if fc[14] >= a.min_score and max(fc[2], fc[3]) >= a.min_face]
        for face in (good[:a.max_per_image] if a.all_faces else good[:1]):
          x, y, w, h = face[:4]
          m = max(w, h) * 0.9
          cx, cy = x + w / 2, y + h / 2
          H, W = img.shape[:2]
          x0, y0 = int(max(0, cx - m)), int(max(0, cy - m * 1.05))
          x1, y1 = int(min(W, cx + m)), int(min(H, cy + m * 0.95))
          crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
          t = Image.fromarray(crop).resize((S, S), Image.LANCZOS)
          n = len(tiles) + 1
          d = ImageDraw.Draw(t)
          d.rectangle([0, 0, S // 3.2, S // 7], fill=(0, 0, 0))
          d.text((4, 1), str(n), font=font, fill=(255, 220, 60))
          tiles.append(t); index.append({"n": n, "file": f, "face_px": int(max(w, h)), "box": f"{int(x)},{int(y)},{int(w)},{int(h)}"})
    cols = a.cols
    anc = Image.open(a.anchor).convert("RGB"); anc.thumbnail((cols * (S + 4) // 2, S * 2))
    aimg = cv2.imread(a.anchor); af = fx.detect(aimg)[0]; ax, ay, aw, ah = af[:4]; am = max(aw, ah) * 0.9
    acrop = cv2.cvtColor(aimg[int(max(0, ay + ah / 2 - am * 1.05)):int(ay + ah / 2 + am * 0.95), int(max(0, ax + aw / 2 - am)):int(ax + aw / 2 + am)], cv2.COLOR_BGR2RGB)
    aface = Image.fromarray(acrop).resize((anc.height, anc.height), Image.LANCZOS)
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (S + 4) + 4, anc.height + 50 + rows * (S + 4) + 4), (22, 22, 26))
    sheet.paste(anc, (4, 4)); sheet.paste(aface, (anc.width + 12, 4))
    ImageDraw.Draw(sheet).text((8, anc.height + 12), "ANCHOR above. For each number: same person as the anchor?",
                               font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22), fill=(235, 235, 235))
    for i, t in enumerate(tiles):
        sheet.paste(t, (4 + (i % cols) * (S + 4), anc.height + 50 + (i // cols) * (S + 4)))
    sheet.save(os.path.join(a.out, "label_sheet.jpg"), quality=90)
    with open(os.path.join(a.out, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "file", "face_px", "box"]); w.writeheader(); w.writerows(index)
    print(f"{len(tiles)} faces -> {a.out}/label_sheet.jpg")


def auc(pos, neg):
    """Mann-Whitney AUC: chance that a random 'yes' scores above a random 'no'."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cmd_validate(a):
    scores = {r["file"]: r for r in csv.DictReader(open(a.scores, encoding="utf-8"))}
    labels = {r["file"]: r["label"].strip().lower() for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
    for key in ("aura", "sface"):
        pos = [float(scores[f][key]) for f, l in labels.items() if l == "yes" and f in scores and scores[f][key] != ""]
        neg = [float(scores[f][key]) for f, l in labels.items() if l == "no" and f in scores and scores[f][key] != ""]
        best = max(((sum(p >= t for p in pos) + sum(n < t for n in neg)) / max(1, len(pos) + len(neg)), t)
                   for t in sorted(set(pos + neg))) if pos and neg else (float("nan"), float("nan"))
        print(f"{key}: AUC {auc(pos, neg):.3f}  yes n={len(pos)} mean {np.mean(pos) if pos else float('nan'):.3f}  "
              f"no n={len(neg)} mean {np.mean(neg) if neg else float('nan'):.3f}  best threshold {best[1]:.3f} (accuracy {best[0]:.2f})")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("--anchor", nargs="+", required=True); s.add_argument("--images", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--pick", choices=["biggest", "left"], default="biggest", help="which face to score when there are several")
    s = sp.add_parser("sheet"); s.add_argument("--anchor", required=True); s.add_argument("--images", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, default=160); s.add_argument("--cols", type=int, default=10)
    s.add_argument("--all-faces", action="store_true"); s.add_argument("--min-face", type=int, default=40)
    s.add_argument("--min-score", type=float, default=0.85); s.add_argument("--max-per-image", type=int, default=2)
    s = sp.add_parser("validate"); s.add_argument("--scores", required=True); s.add_argument("--labels", required=True)
    a = ap.parse_args()
    {"score": cmd_score, "sheet": cmd_sheet, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    main()
