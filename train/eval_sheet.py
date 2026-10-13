"""Evaluation sheets for a character LoRA: per scene, the no-LoRA baseline next to each checkpoint, with face scores.

  python train/eval_sheet.py --dir train/pilot/eval_v0 --steps 500,1000,1500,2000 --anchors A0.png A1.png --out train/pilot/aidan/review
  python train/eval_sheet.py --dir train/pilot/eval_v2 --cols base,r1500,c1500 --label "r1500=v1 1500" --front e01_tavern,e02_rain --left e08_withcain ...
Face scores land in <dir>/face_scores.csv (biggest face) and, for --left scenes, <dir>/face_scores_left.csv (left-most face).
"""
import argparse, csv, glob, os, statistics as st, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--steps", default=""); ap.add_argument("--anchors", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--name", default="v0"); ap.add_argument("--thr", type=float, default=0.38)
ap.add_argument("--cols", default="", help="column codes in order (default: base, s<steps>, c<steps>)")
ap.add_argument("--label", action="append", default=[], help="CODE=header text")
ap.add_argument("--front", default="", help="fixed face scenes for the face bar (a missing face counts as a miss)")
ap.add_argument("--left", default="", help="two-shot scenes: score the left-most face instead of the biggest")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
steps = [s for s in a.steps.split(",") if s]
N = lambda p: p.replace(chr(92), "/")
ids = [sys.executable, "tools/idscore.py", "score", "--anchor", *a.anchors]
scores_csv = os.path.join(a.dir, "face_scores.csv")
subprocess.run(ids + ["--images", f"{a.dir}/*/*.png", "--out", scores_csv], check=True, capture_output=True)
sc = {N(r["file"]): r for r in csv.DictReader(open(scores_csv, encoding="utf-8"))}
if a.left:
    left_csv = os.path.join(a.dir, "face_scores_left.csv")
    subprocess.run(ids + ["--pick", "left", "--images", *[f"{a.dir}/{s}_*/*.png" for s in a.left.split(",")], "--out", left_csv], check=True, capture_output=True)
    sc.update({N(r["file"]): r for r in csv.DictReader(open(left_csv, encoding="utf-8"))})
dirs = [d for d in glob.glob(f"{a.dir}/*") if os.path.isdir(d)]
scenes = sorted({os.path.basename(d).rsplit("_", 1)[0] for d in dirs})
cols = a.cols.split(",") if a.cols else ["base"] + [f"s{s}" for s in steps] + [f"c{s}" for s in steps if glob.glob(f"{a.dir}/*_c{s}")]
labels = dict(x.split("=", 1) for x in a.label)
head = lambda c: labels.get(c) or ("baseline (text block)" if c == "base" else (f"step {c[1:]}" if c[0] == "s" else f"step {c[1:]} + costume"))
front = set(filter(None, a.front.split(",")))
F = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18); FB = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
TW, TH = 300, 169
per = {c: [] for c in cols}; fr = {c: [0, 0] for c in cols}
for seed_i in range(max(len(glob.glob(f"{d}/*.png")) for d in dirs)):
    W = 170 + len(cols) * (TW + 6); H = 76 + len(scenes) * (TH + 8)
    s = Image.new("RGB", (W, H), (22, 22, 26)); d = ImageDraw.Draw(s)
    d.text((10, 10), f"LoRA {a.name}: no-LoRA baseline vs checkpoints (seed set {seed_i + 1}); yellow = face score vs anchor", font=FB, fill=(235, 235, 235))
    for j, c in enumerate(cols):
        d.text((170 + j * (TW + 6), 46), head(c), font=F, fill=(170, 220, 170))
    for i, scn in enumerate(scenes):
        y = 76 + i * (TH + 8)
        d.text((8, y + 8), scn, font=F, fill=(235, 235, 235))
        for j, c in enumerate(cols):
            files = sorted(glob.glob(f"{a.dir}/{scn}_{c}/*.png"))
            if len(files) <= seed_i:
                continue
            f = N(files[seed_i])
            im = Image.open(f).convert("RGB"); im.thumbnail((TW, TH)); x = 170 + j * (TW + 6); s.paste(im, (x, y))
            v = sc.get(f, {}).get("aura", "")
            if v:
                per[c].append(float(v))
            if scn in front:
                fr[c][0] += 1; fr[c][1] += bool(v) and float(v) >= a.thr
            d.rectangle([x, y, x + 70, y + 22], fill=(0, 0, 0)); d.text((x + 4, y + 2), v[:4] if v else "-", font=F, fill=(250, 210, 70))
    s.save(os.path.join(a.out, f"3_eval_{a.name}_seed{seed_i + 1}.jpg"), quality=86)
print(f"face score per column (all scenes, all seeds; share >= {a.thr}" + (f"; face scenes = {a.front}" if front else "") + "):")
for c in cols:
    v = per[c]
    print(f"  {c:6s} n={len(v):2d} mean={st.mean(v) if v else float('nan'):.3f} median={st.median(v) if v else float('nan'):.3f} "
          f"share: {sum(x >= a.thr for x in v)}/{len(v)}" + (f"  face scenes: {fr[c][1]}/{fr[c][0]}" if front else ""))
