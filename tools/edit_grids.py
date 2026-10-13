"""Labelled grids for judging edits and re-render candidates (2026-10-12).
Reads a spec JSON {"items": [{"key", "input" (optional original), "cands": [glob, ...], "check"}, ...]} and writes
<out>/<key>.jpg: the original first (labelled INPUT, if given) then each candidate labelled A, B, C ... in a grid of 3 columns,
plus <out>/<key>.json mapping each label to its file. Items with no candidate files yet are skipped and listed.
  python tools/edit_grids.py shots/ep01_v3_edits/judge_spec.json shots/ep01_v3_edits/grids
"""
import glob, json, os, sys
import cv2, numpy as np

spec, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
W, H = 640, 360
missing = []
for it in json.load(open(spec, encoding="utf-8"))["items"]:
    files = []
    for g in it["cands"]:
        files += sorted(glob.glob(g))
    if not files:
        missing.append(it["key"]); continue
    tiles, labels = [], {}
    if it.get("input"):
        im = cv2.imread(it["input"]); t = cv2.resize(im, (W, H))
        cv2.putText(t, "INPUT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3); tiles.append(t)
    for i, f in enumerate(files):
        lab = "ABCDEFGHIJKLMNOP"[i]; labels[lab] = f.replace("\\", "/")
        t = cv2.resize(cv2.imread(f), (W, H))
        cv2.putText(t, lab, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3); tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros((H, W, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(out, f"{it['key']}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump({"labels": labels, "input": it.get("input"), "check": it["check"]}, open(os.path.join(out, f"{it['key']}.json"), "w", encoding="utf-8"), indent=1)
print("grids done; missing:", missing)
