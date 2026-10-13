"""Prepare the next clip-QC round (2026-10-13): find clips finished in a video batch log that no qc_*.json judged yet,
build their frame sheets (tools/clip_sheets.py) and info files with the motion prompt the batch used, and print the ids as JSON.
  python tools/qc_prep.py logs/video_batch2.out story/diablo/episodes/ep01_the_butcher/shots_ep01_video_batch2.csv
"""
import csv, glob, json, os, re, subprocess, sys

log, batch = sys.argv[1], sys.argv[2]
Q = "shots/ep01_clip_qc"
done = re.findall(r"\[(b\d\d_\d\d\w?)\] take .*-> (\S+)", open(log, encoding="utf-8").read())
judged = {}
for f in glob.glob(f"{Q}/qc_*.json"):
    for r in json.load(open(f, encoding="utf-8")):
        judged.setdefault(r["id"], set()).add(f)
mp = {r["id"]: r["motion_prompt"] for r in csv.DictReader(open(batch, encoding="utf-8"))}
# a clip counts as judged only if a qc file written after the clip exists for it
new = []
for k, clip in done:
    clip = clip.replace("\\", "/")
    t = os.path.getmtime(clip)
    if not any(os.path.getmtime(f) > t for f in judged.get(k, ())):
        new.append(k)
new = list(dict.fromkeys(new))
if new:
    subprocess.run([sys.executable, "tools/clip_sheets.py", "--ids", ",".join(new)], check=True, stdout=subprocess.DEVNULL)
    idx = json.load(open(f"{Q}/index.json", encoding="utf-8"))
    for k in new:
        v = idx[k]
        json.dump({"id": k, "dur": v["dur"], "framing": v["framing"], "who": v["who"], "motion": mp.get(k, v["motion"]), "lines": v["lines"],
                   "seconds": v["seconds"], "note": "The clip was generated with length = dur + 0.5 s. Light-only effects (glows, fades) may be added later in the grade."},
                  open(f"{Q}/info/{k}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(new))
