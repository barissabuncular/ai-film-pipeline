"""Gated skin-texture pass over a character dataset (made the standard step for human LoRAs on 2026-10-06, withdrawn on
2026-10-07: a blind visual check found the full pass drifts identity, which the face score missed; see train/texture_composite.py).
Why: the Qwen edit model leaves airbrushed skin and a fingerprint/crosshatch mesh; Moreina v1 learned that gloss. The texture test
(train/texture_test.py, train/pilot/texture_test/card_texture.jpg) showed a Z-Image img2img pass at 0.15-0.25 removes the mesh and gives
real pores, but on some images it shifts the face or the expression, so each image is gated by its face score.

  python train/texture_pass.py train/pilot/<char>/texture.json      (ComfyUI must be up; run inside train/edit_session.py as a py: job)
Config: {"src": "train/datasets/<x>", "out": "train/datasets/<y>", "anchors": [A0, A1], "trigger": "dkcn man",
         "subject": "plain words for the person (Z-Image does not know the trigger)", "max_drop": 0.03}
Per image: img2img at 0.25 and 0.15 (prompt = style + subject + the caption without the trigger); AuraFace vs the anchors; take 0.25 if it
scores at least orig - max_drop, else 0.15 if it does, else the original; no face found -> 0.15. Writes <out>/{near,far} with the picks and
the original captions, <out>/dataset.toml (new image and cache paths; caches are rebuilt, so train WITHOUT --skip-cache) and
<out>/texture_pass.csv. Candidates stay in <out>/_cand.
"""
import csv, glob, json, os, shutil, subprocess, sys

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(G)
PY = sys.executable
STYLE = "cinematic film still, photoreal, natural skin texture with pores and fine lines, real light, fine 35mm film grain"
DEN = (0.25, 0.15)
C = json.load(open(sys.argv[1], encoding="utf-8"))
STYLE, DEN = C.get("style", STYLE), tuple(C.get("den", DEN))  # 2026-10-08: whole-image re-render test uses den [0.5] and a live-action style
SRC, OUT, MAXD = C["src"].rstrip("/"), C["out"].rstrip("/"), float(C.get("max_drop", 0.03))
assert os.path.exists(f"{SRC}/dataset.toml"), f"{SRC}/dataset.toml missing (run select_char.py first)"
assert not os.path.exists(f"{OUT}/dataset.toml"), f"{OUT}/dataset.toml exists already; use a new out"
CAND = f"{OUT}/_cand"; os.makedirs(CAND, exist_ok=True)
imgs = sorted(f for sub in ("near", "far") for f in glob.glob(f"{SRC}/{sub}/*.png"))
jobs = []
for i, f in enumerate(imgs):
    sub, name = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4]
    cap = open(f[:-4] + ".txt", encoding="utf-8").read().strip()
    body = cap[len(C["trigger"]):].lstrip(" ,") if cap.startswith(C["trigger"]) else cap
    shutil.copyfile(f, f"{CAND}/{sub}__{name}__orig.png")
    for d in DEN:
        out = f"{CAND}/{sub}__{name}__d{int(d * 100)}.png"
        if not os.path.exists(out):
            jobs.append({"file": f, "prompt": f"{STYLE}, {C['subject']}, {body}", "out": out, "denoise": d, "seed": 9100 + i})
print(f"{len(imgs)} images, {len(jobs)} img2img jobs", flush=True)
if jobs:
    with open(f"{CAND}/jobs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(jobs[0])); w.writeheader(); w.writerows(jobs)
    r = subprocess.run([PY, "-u", "tools/run_img2img.py", "--csv", f"{CAND}/jobs.csv"])
    assert r.returncode == 0, f"run_img2img failed ({r.returncode})"
if C.get("cand_only"):  # 2026-10-08: the AuraFace gate drifts identity; the skin-only composite + visual gate (train/composite_gate.py) decides
    print(f"DONE candidates only -> {CAND}", flush=True); sys.exit(0)
sc_csv = f"{CAND}/scores.csv"
subprocess.run([PY, "tools/idscore.py", "score", "--anchor", *C["anchors"], "--images", f"{CAND}/*.png", "--out", sc_csv], check=True)
sc = {os.path.basename(r["file"]): (float(r["aura"]) if r["aura"] else None) for r in csv.DictReader(open(sc_csv, encoding="utf-8"))}
rows = []
for f in imgs:
    sub, name = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4]
    s = {v: sc.get(f"{sub}__{name}__{v}.png") for v in ("orig", "d25", "d15")}
    if s["orig"] is None:
        pick = "d15" if os.path.exists(f"{CAND}/{sub}__{name}__d15.png") else "orig"; why = "no face found"
    else:
        pick = next((v for v in ("d25", "d15") if s[v] is not None and s[v] >= s["orig"] - MAXD), "orig")
        why = f"orig {s['orig']:.3f}, d25 {s['d25'] if s['d25'] is None else round(s['d25'], 3)}, d15 {s['d15'] if s['d15'] is None else round(s['d15'], 3)}"
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    shutil.copyfile(f"{CAND}/{sub}__{name}__{pick}.png", f"{OUT}/{sub}/{name}.png")
    shutil.copyfile(f[:-4] + ".txt", f"{OUT}/{sub}/{name}.txt")
    rows.append({"image": f"{sub}/{name}.png", "pick": pick, "why": why})
with open(f"{OUT}/texture_pass.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["image", "pick", "why"]); w.writeheader(); w.writerows(rows)
for sub in ("near", "far"):
    if os.path.isdir(f"{SRC}/{sub}"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
toml = open(f"{SRC}/dataset.toml", encoding="utf-8").read()
a, b = os.path.abspath(SRC).replace("\\", "/"), os.path.abspath(OUT).replace("\\", "/")
assert a in toml, "source path not found in dataset.toml"
open(f"{OUT}/dataset.toml", "w", encoding="utf-8").write(toml.replace(a, b).replace("# ", f"# texture pass of {SRC} (train/texture_pass.py); ", 1))
n = {v: sum(r["pick"] == v for r in rows) for v in ("d25", "d15", "orig")}
print(f"DONE texture pass {SRC} -> {OUT}: {n}", flush=True)
