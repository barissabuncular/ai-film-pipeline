"""Texture fix test for edit-model outputs (the Qwen-Image-Edit airbrushed skin + checker mesh that made Moreina v1 glossy).
Variants per image: original, Z-Image img2img at denoise 0.15 / 0.25 / 0.35 (photoreal texture prompt), and CPU film grain + mild unsharp.
Gate: AuraFace drop against the character's anchors (A0+A1); Aidan's 2026-10-02 harmonise test lost 0.24 at 0.20.
  python train/texture_test.py build      (CPU: picks images, writes the img2img job CSV and the grain variants)
  python tools/run_img2img.py --csv train/pilot/texture_test/jobs.csv      (GPU, ComfyUI up)
  python train/texture_test.py score      (CPU: face scores per variant -> train/pilot/texture_test/scores.csv + sheet)
"""
import csv, glob, os, subprocess, sys
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

OUT = "train/pilot/texture_test"; PY = sys.executable
SETS = {  # character: (anchors, images, subject words for the prompt)
    "moreina": (["train/pilot/moreina/A0.png", "train/pilot/moreina/A1.png"], sorted(glob.glob("train/datasets/morvn_v1/near/*.png"))[:6],
                "a woman of about 28 with loose long wavy auburn-red hair, black leather armour trimmed with silver"),
    "cain": (["train/pilot/cain/A0.png", "train/pilot/cain/A1.png"],
             [f for f in sorted(glob.glob("train/pilot/cain/w1/*/ed_*.png")) if "/fix_" not in f.replace("\\", "/")][::5][:6],
             "an old bald man with a short white beard and a grey-blue wool robe"),
    # anchors whose own skin failed the w1 judges (2026-10-06): scored against the approved scene face, so a fix may even raise the score
    "farnham": (["train/pilot/farnham/scene_face.png"], ["train/pilot/farnham/A0.png", "train/pilot/farnham/A1.png"],
                "a heavy-set drunk of about 50 with a puffy ruddy face, lank greasy grey hair to the jaw and a straggly grey-brown beard, a filthy torn grey-olive wool tunic"),
    "ogden": (["train/pilot/ogden/scene_face.png"], ["train/pilot/ogden/A0.png"],
              "a lanky clean-shaven innkeeper in his late thirties with a long thin face, hollow cheeks and thick straight copper-red hair with a short fringe, a rust-red wool shoulder cape and a dark brown leather vest"),
    "lazarus": (["train/pilot/lazarus/scene_face.png"], ["train/pilot/lazarus/A0.png", "train/pilot/lazarus/A1.png"],
                "a priest of about 45 with short dark brown hair combed back and a broad square clean-shaven face, long ivory robes with a mustard-gold collar band"),
}
DEN = (0.15, 0.25, 0.35)
STYLE = "cinematic film still, photoreal, natural skin texture with pores and fine lines, real light, fine 35mm film grain"


def grain(im, seed):
    rng = np.random.default_rng(seed); a = np.asarray(im.convert("RGB")).astype(np.float32)
    lum = rng.normal(0, 7.0, a.shape[:2])[..., None]  # luminance grain, about 3% of range
    out = np.clip(a + lum, 0, 255).astype(np.uint8)
    return Image.fromarray(out).filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))


if sys.argv[1] == "build":
    os.makedirs(OUT, exist_ok=True); rows = []
    for ch, (anch, files, subj) in SETS.items():
        for i, f in enumerate(files):
            tag = f"{ch}_{i:02d}"; Image.open(f).convert("RGB").save(f"{OUT}/{tag}_orig.png")
            grain(Image.open(f), i).save(f"{OUT}/{tag}_grain.png")
            for d in DEN:
                rows.append({"file": f, "prompt": f"{STYLE}, {subj}", "out": f"{OUT}/{tag}_d{int(d * 100)}.png", "denoise": d, "seed": 8800 + i})
    with open(f"{OUT}/jobs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "img2img jobs;", sum(len(v[1]) for v in SETS.values()), "images")
else:
    F = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18); out = []
    for ch, (anch, files, subj) in SETS.items():
        sc_csv = f"{OUT}/{ch}_scores.csv"
        subprocess.run([PY, "tools/idscore.py", "score", "--anchor", *anch, "--images", f"{OUT}/{ch}_*.png", "--out", sc_csv], capture_output=True)
        sc = {os.path.basename(r["file"]): r["aura"] for r in csv.DictReader(open(sc_csv, encoding="utf-8"))}
        vars_ = ["orig", "grain"] + [f"d{int(d * 100)}" for d in DEN]
        sheet = Image.new("RGB", (len(vars_) * 330, len(files) * 360 + 30), (20, 20, 20)); dr = ImageDraw.Draw(sheet)
        for j, v in enumerate(vars_):
            dr.text((j * 330 + 4, 4), v, font=F, fill=(240, 200, 80))
        for i in range(len(files)):
            for j, v in enumerate(vars_):
                p = f"{OUT}/{ch}_{i:02d}_{v}.png"
                if os.path.exists(p):
                    im = Image.open(p).convert("RGB"); im.thumbnail((320, 320)); sheet.paste(im, (j * 330, 30 + i * 360))
                    dr.text((j * 330 + 4, 30 + i * 360 + 322), f"face {sc.get(os.path.basename(p), '-')}", font=F, fill=(250, 210, 70))
                    out.append({"char": ch, "img": i, "variant": v, "aura": sc.get(os.path.basename(p), "")})
        sheet.save(f"{OUT}/sheet_{ch}.jpg", quality=88)
    with open(f"{OUT}/scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["char", "img", "variant", "aura"]); w.writeheader(); w.writerows(out)
    for ch in SETS:
        for v in ["orig", "grain"] + [f"d{int(d * 100)}" for d in DEN]:
            xs = [float(r["aura"]) for r in out if r["char"] == ch and r["variant"] == v and r["aura"]]
            print(f"{ch:8s} {v:6s} mean face {sum(xs) / max(1, len(xs)):.3f} (n={len(xs)})")
