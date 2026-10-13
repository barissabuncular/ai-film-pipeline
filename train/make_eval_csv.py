"""Write the checkpoint evaluation CSV for run_shots.py (keyframe_zimage_lora.json): every eval scene x seed x checkpoint,
plus a no-LoRA baseline that uses the character's text block instead of the trigger.

  python train/make_eval_csv.py --name aidnw_v0 --trigger "aidnw man" --block AIDAN --steps 500,1000,1500,2000 --seeds 2 --out train/pilot/eval_v0.csv
Free columns: --col CODE=LORA@STRENGTH:SUBJECT (repeatable), SUBJECT = trigger | costume | close, LORA without .safetensors.
  --col r1500=aidnw_v1_s1500@1.0:costume   --col k1500=aidnw_v2_s1500@1.0:close --close "in a ..." --close-scenes e01_tavern,e02_rain
"""
import argparse, csv, re

ap = argparse.ArgumentParser()
ap.add_argument("--name", default=""); ap.add_argument("--trigger", required=True); ap.add_argument("--block", required=True)
ap.add_argument("--steps", default=""); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--out", required=True)
ap.add_argument("--prompts", default="train/pilot/eval_prompts.csv"); ap.add_argument("--strength", default="1.0")
ap.add_argument("--costume", default="", help="short costume phrase; adds a LoRA+costume column for every checkpoint")
ap.add_argument("--col", action="append", default=[], help="CODE=LORA@STRENGTH:SUBJECT")
ap.add_argument("--close", default="", help="phrase after the trigger for 'close' columns")
ap.add_argument("--close-scenes", default="", help="scene ids that get the 'close' columns (default: all)")
ap.add_argument("--no-base", action="store_true")
ap.add_argument("--block-text", default="", help="baseline subject text instead of the visual bible block")
a = ap.parse_args()
cols = [re.fullmatch(r"(\w+)=([\w.-]+)@([\d.]+):(trigger|costume|close)", c).groups() for c in a.col]
subj = {"trigger": a.trigger, "costume": f"{a.trigger} {a.costume}", "close": f"{a.trigger} {a.close}"}
bible = open("story/diablo/look/visual_bible.md", encoding="utf-8").read()
block = a.block_text or re.search(rf"^- Block `{a.block}`: (.+)$", bible, re.M).group(1).strip()
rows = []
for p in csv.DictReader(open(a.prompts, encoding="utf-8")):
    for s in range(a.seeds):
        seed = 90000 + int(p["id"][1:3]) * 10 + s
        for step in filter(None, a.steps.split(",")):
            rows.append({"id": f"{p['id']}_s{step}", "image_prompt": p["prompt"].replace("{S}", a.trigger), "motion_prompt": "", "seed": seed,
                         "width": 1280, "height": 720, "lora": f"{a.name}_s{step}.safetensors", "lora_strength": a.strength})
            if a.costume:
                rows.append({"id": f"{p['id']}_c{step}", "image_prompt": p["prompt"].replace("{S}", f"{a.trigger} {a.costume}"), "motion_prompt": "", "seed": seed,
                             "width": 1280, "height": 720, "lora": f"{a.name}_s{step}.safetensors", "lora_strength": a.strength})
        for code, lora, strength, kind in cols:
            if kind == "close" and a.close_scenes and p["id"] not in a.close_scenes.split(","):
                continue
            rows.append({"id": f"{p['id']}_{code}", "image_prompt": p["prompt"].replace("{S}", subj[kind]), "motion_prompt": "", "seed": seed,
                         "width": 1280, "height": 720, "lora": "none" if lora == "none" else f"{lora}.safetensors", "lora_strength": strength})
        if not a.no_base:
            rows.append({"id": f"{p['id']}_base", "image_prompt": p["prompt"].replace("{S}", block), "motion_prompt": "", "seed": seed,
                         "width": 1280, "height": 720, "lora": "none", "lora_strength": ""})
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "image_prompt", "motion_prompt", "seed", "width", "height", "lora", "lora_strength"]); w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", a.out)
