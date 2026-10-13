"""Build the SeedVR2 post manifest (2026-10-13): for every shot, pick the approved clip and its usable span from the clip-QC results.
QC files and the clip seeds they judged: qc_1/qc_2 = batch 1 (clip_31*), qc_b2_* = batch 2 (clip_34*), qc_b3_* = batch 3 (clip_36*), qc_b4_* = batch 4 (clip_37*).
A shot's pick is its newest clip whose QC verdict is pass or trim; pass beats trim within the same batch. Writes shots/ep01_post/manifest.json:
{id: {clip, from, to, dur, verdict, qc}} and lists shots with no usable clip yet.
  python tools/post_manifest.py
"""
import glob, json, os, re

Q = "shots/ep01_clip_qc"
SEED = {"qc_1": "31", "qc_2": "31", "qc_b2": "34", "qc_b3": "36", "qc_b4": "37", "qc_b5": "38"}
master = json.load(open("shots/ep01_master.json", encoding="utf-8"))
EX = set(json.load(open("shots/ep01_post/exclude.json", encoding="utf-8"))["clips"]) if os.path.exists("shots/ep01_post/exclude.json") else set()
best = {}
for f in sorted(glob.glob(f"{Q}/qc_*.json")):
    name = os.path.basename(f)[:-5]
    pre = next((v for k, v in SEED.items() if name == k or name.startswith(k + "_")), None)
    if pre is None:
        continue
    for r in json.load(open(f, encoding="utf-8")):
        if r["final"] not in ("pass", "trim"):
            continue
        clips = [c for c in sorted(glob.glob(f"shots/ep01_v1/{r['id']}/clip_{pre}*.mp4")) if c.replace("\\", "/") not in EX]
        if not clips:
            continue
        rank = (int(pre), 1 if r["final"] == "pass" else 0)
        if r["id"] not in best or rank > best[r["id"]]["rank"]:
            best[r["id"]] = {"rank": rank, "clip": clips[-1].replace("\\", "/"), "from": r["usable_from"], "to": r["usable_to"],
                             "dur": master.get(r["id"], {}).get("dur"), "verdict": r["final"], "qc": name}
for v in best.values():
    v.pop("rank")
os.makedirs("shots/ep01_post", exist_ok=True)
json.dump(best, open("shots/ep01_post/manifest.json", "w", encoding="utf-8"), indent=1)
need = [k for k, v in master.items() if v["status"] == "ok" and k not in best]
print(len(best), "shots with an approved clip;", len(need), "ok shots without one:", need)
