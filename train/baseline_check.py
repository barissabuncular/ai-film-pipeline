"""Text-only baseline check before spending LoRA GPU hours (2026-10-07). Jazreth's and Cain's text blocks beat their LoRAs (27/36 and
32/36 on model); so each character's no-LoRA baseline (the 'base' rows of its eval CSV: 12 scenes x 3 seeds) is rendered and judged
blind first. Renders go to train/pilot/eval_<lora name>/, the folder night_char.py uses later, so they are not rendered twice.

  python train/baseline_check.py render CFG.json     (ComfyUI up; run as 'py:train/baseline_check.py@CFG.json' in train/edit_session.py)
  python train/baseline_check.py blind CFG.json      (CPU: blind copies train/pilot/eval_<name>_base_blind, code X01, for judge_eval_char.js)
CFG: [{"char": "lazarus", "name": "lzrs_v1", "eval_csv": "train/pilot/lazarus/eval_lzrs_v1.csv", "anchors": [A0, A1]}, ...]
Rule (fixed 2026-10-07 before any baseline render): a character whose baseline is on model (identity 6+ and costume 7+) in at least
27 of 36 images AND scores photoreal 7+ is proposed to the user as 'text block, no LoRA'; the others go to LoRA training.
"""
import csv, json, os, shutil, subprocess, sys, time
from PIL import Image

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(G)
PY = sys.executable
args = sys.argv[1:] if len(sys.argv) > 2 else ["render"] + sys.argv[1:]  # a py: job in edit_session.py passes only the config
mode, cfg = args[0], json.load(open(args[1], encoding="utf-8"))
for c in cfg:
    rows = [r for r in csv.DictReader(open(c["eval_csv"], encoding="utf-8")) if r["lora"] == "none"]
    out = f"train/pilot/eval_{c['name']}"
    if mode == "render":
        todo = [r for r in rows if not os.path.exists(f"{out}/{r['id']}/kf_{r['seed']}.png")]
        if not todo:
            print(f"{c['char']}: {len(rows)} baseline images already there", flush=True); continue
        tmp = f"logs/base_{c['name']}_todo.csv"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(todo)
        t0 = time.time()
        rc = subprocess.run([PY, "-u", "run_shots.py", "keyframes", "--kf-workflow", "keyframe_zimage_lora.json", "--shots", tmp, "--out-dir", out,
                             "--candidates", "1", "--poll", "1", "--keep-going"], stdout=open(f"logs/base_{c['name']}.log", "a", encoding="utf-8"),
                            stderr=subprocess.STDOUT).returncode
        k = sum(os.path.exists(f"{out}/{r['id']}/kf_{r['seed']}.png") for r in rows)
        print(f"{c['char']}: {k}/{len(rows)} baseline images, exit {rc}, {(time.time() - t0) / max(1, len(todo)):.1f} s/image", flush=True)
    else:
        dest = f"train/pilot/eval_{c['name']}_base_blind"; shutil.rmtree(dest, ignore_errors=True); os.makedirs(f"{dest}/_ref")
        for a in c["anchors"]:
            shutil.copyfile(a, f"{dest}/_ref/{os.path.basename(a)}")
        info = {}
        for r in rows:
            scn = r["id"].rsplit("_", 1)[0]; f = f"{out}/{r['id']}/kf_{r['seed']}.png"
            if os.path.exists(f):
                os.makedirs(f"{dest}/{scn}_X01", exist_ok=True)
                im = Image.open(f); im.load(); im.save(f"{dest}/{scn}_X01/kf_{r['seed']}.png")  # drops ComfyUI metadata
                info.setdefault(scn, {})[f"{scn}_X01/kf_{r['seed']}.png"] = r["image_prompt"]
        for scn, d in info.items():
            json.dump(d, open(f"{dest}/prompts_{scn}.json", "w", encoding="utf-8"), indent=1)
        print(f"{c['char']}: {sum(len(d) for d in info.values())} images in {dest}; scenes {sorted(info)}", flush=True)
print("DONE", flush=True)
