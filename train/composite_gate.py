"""Skin-only composite + blind visual gate for one character dataset (the method from Moreina's gate 2, 2026-10-08).
  1. python train/texture_pass.py train/pilot/<char>/skin.json        (ComfyUI up; config has "cand_only": true -> <out>/_cand)
  2. python train/composite_gate.py blind train/pilot/<char>/skin.json  (CPU: train/texture_composite.py + shuffled blind copies)
  3. Workflow skin-pass-visual-gate on <out>/_blind (2 judges per image) -> save the result list as <out>/gate_judge.json
  4. python train/composite_gate.py build train/pilot/<char>/skin.json  (CPU: picks by the fixed rule, writes <out> as a dataset)
Rule (fixed for Moreina before her gate): a composite (c15/c25) is used only if both judges call it the same person (yes, or yes+close),
neither reports anything important lost, and its summed skin rating beats the original's; the better skin wins, tie -> c15; else original.
Config (skin.json): {"src": "train/datasets/<x>", "out": "train/datasets/<x>c", "trigger": ..., "subject": ..., "cand_only": true}
"""
import collections, csv, json, os, shutil, subprocess, sys
from PIL import Image

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(G)
mode, C = sys.argv[1], json.load(open(sys.argv[2], encoding="utf-8"))
SRC, OUT = C["src"].rstrip("/"), C["out"].rstrip("/"); CAND, COMP, BL = f"{OUT}/_cand", f"{OUT}/_comp", f"{OUT}/_blind"
KEY = f"{OUT}/_gate_key.json"
SK = {"natural, real pores": 3, "slightly smooth": 2, "airbrushed or waxy": 1, "face too small to tell": 0}
if mode == "blind":
    subprocess.run([sys.executable, "train/texture_composite.py", CAND, COMP], check=True)
    stems = sorted(os.path.basename(p)[:-len("__orig.png")] for p in os.listdir(COMP) if p.endswith("__orig.png"))
    perms = [("orig", "c15", "c25"), ("c25", "orig", "c15"), ("c15", "c25", "orig"), ("orig", "c25", "c15"), ("c25", "c15", "orig"), ("c15", "orig", "c25")]
    shutil.rmtree(BL, ignore_errors=True); os.makedirs(BL); key = {}
    for i, s in enumerate(stems):
        for j in (0, 1):
            for lab, v in zip("ABC", perms[(i * 2 + j) % 6]):
                Image.open(f"{COMP}/{s}__{v}.png").convert("RGB").save(f"{BL}/{i:02d}_j{j}_{lab}.png"); key[f"{i:02d}_j{j}_{lab}"] = v
        key[f"{i:02d}"] = s
    json.dump(key, open(KEY, "w"), indent=1); print(f"{len(stems)} images -> {BL} (n={len(stems)} for the gate workflow)")
else:
    key = json.load(open(KEY)); res = json.load(open(f"{OUT}/gate_judge.json", encoding="utf-8"))
    V = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in res:
        for v in r["versions"]:
            V[r["img"]][key[f"{r['img']}_j{r['judge']}_{v['label']}"]].append(v)
    rows = []
    for img, vv in sorted(V.items()):
        stem = key[img]; sub, name = stem.split("__", 1); ok = []
        for var in ("c15", "c25"):
            vs = vv[var]; sp = sorted(x["same_person"] for x in vs)
            if (sp in (["yes", "yes"], ["close", "yes"]) or all(x["same_person"] == "face not visible" for x in vs)) and not any(x["lost"].strip() for x in vs):
                ok.append((sum(SK[x["skin"]] for x in vs), 1 if var == "c15" else 0, var))
        o = sum(SK[x["skin"]] for x in vv["orig"]); b = max(ok) if ok else None; pick = b[2] if b and b[0] > o else "orig"
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
        shutil.copyfile(f"{COMP}/{stem}__{pick}.png" if pick != "orig" else f"{SRC}/{sub}/{name}.png", f"{OUT}/{sub}/{name}.png")
        shutil.copyfile(f"{SRC}/{sub}/{name}.txt", f"{OUT}/{sub}/{name}.txt"); rows.append({"image": f"{sub}/{name}.png", "pick": pick})
    for sub in ("near", "far"):
        if os.path.isdir(f"{SRC}/{sub}"):
            os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    a, b = os.path.abspath(SRC).replace("\\", "/"), os.path.abspath(OUT).replace("\\", "/")
    toml = open(f"{SRC}/dataset.toml", encoding="utf-8").read(); assert a in toml
    open(f"{OUT}/dataset.toml", "w", encoding="utf-8").write(toml.replace(a, b).replace("# ", f"# skin-only composite of {SRC} (train/composite_gate.py); ", 1))
    with open(f"{OUT}/picks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "pick"]); w.writeheader(); w.writerows(rows)
    print(dict(collections.Counter(r["pick"] for r in rows)), "->", OUT)
