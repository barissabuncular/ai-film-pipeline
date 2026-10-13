"""Save a clip-QC workflow result and update the retake queue (2026-10-13).
  python tools/qc_save.py <workflow output file> <name, e.g. qc_b2_2> [batch tag]
Writes shots/ep01_clip_qc/<name>.json, adds every 'retake' to shots/ep01_clip_qc/retake_queue.json (removes ids that now pass or trim),
and prints a one-line tally plus the retakes.
"""
import json, os, sys

src, name = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else name
Q = "shots/ep01_clip_qc"
d = json.load(open(src, encoding="utf-8")); r = d["result"]
json.dump(r, open(f"{Q}/{name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
p = f"{Q}/retake_queue.json"; q = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
for x in r:
    v = x.get("verify") or {}
    if x["final"] == "retake":
        q[x["id"]] = {"from": tag, "faults": [f["what"] for f in v.get("faults", []) if f["severity"] == "major"], "judge_motion": v.get("retake_motion", "")}
    else:
        q.pop(x["id"], None)
json.dump(q, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
c = {k: sum(1 for x in r if x["final"] == k) for k in ("pass", "trim", "retake")}
print(name, c, "| retakes:", [x["id"] for x in r if x["final"] == "retake"], "| queue size", len(q))
