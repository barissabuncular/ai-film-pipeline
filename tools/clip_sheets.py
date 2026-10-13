"""Frame sheets for clip QC (2026-10-12): for each shot with a clip, write <out>/<id>.jpg = the keyframe plus 8 frames of the clip
(0, 0.7, 1.4, ... 4.2 s and the last frame, at 16 fps) in a 3x3 grid with time stamps, and <out>/index.json with the shot's framing, who, motion prompt,
duration and the clip path, for the clip-QC judges. Skips sheets that already exist.
  python tools/clip_sheets.py --shots-dir shots/ep01_v1 --out shots/ep01_clip_qc [--ids b19_01,b19_02]
"""
import argparse, glob, json, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shots-dir", default="shots/ep01_v1"); ap.add_argument("--out", default="shots/ep01_clip_qc")
ap.add_argument("--ids", default="")
ap.add_argument("--shot-list", default="story/diablo/episodes/ep01_the_butcher/shot_list_v3.json")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sl = json.load(open(a.shot_list, encoding="utf-8")); by = {s["id"]: s for s in (sl["shots"] if isinstance(sl, dict) else sl)}
idx_path = os.path.join(a.out, "index.json")
idx = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else {}
ids = a.ids.split(",") if a.ids else sorted(os.listdir(a.shots_dir))
W, H = 640, 362
for k in ids:
    clips = sorted(glob.glob(os.path.join(a.shots_dir, k, "clip_*.mp4")))
    if not clips or k not in by:
        continue
    clip = clips[-1].replace("\\", "/"); sheet = os.path.join(a.out, f"{k}.jpg")
    if os.path.exists(sheet) and idx.get(k, {}).get("clip") == clip and idx[k].get("v") == 2:
        continue
    cap = cv2.VideoCapture(clip); fps = cap.get(cv2.CAP_PROP_FPS) or 16; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    key = cv2.imread(os.path.join(a.shots_dir, k, "selected.png"))
    t = cv2.resize(key, (W, H)); cv2.putText(t, "KEYFRAME", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2); tiles.append(t)
    for i in range(8):
        f_i = n - 1 if i == 7 else min(n - 1, int(round(i * 0.7 * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_i); ok, f = cap.read()
        t = cv2.resize(f, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
        cv2.putText(t, f"{f_i / fps:.1f}s", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2); tiles.append(t)
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
    cv2.imwrite(sheet, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    s = by[k]
    idx[k] = dict(clip=clip, sheet=sheet.replace("\\", "/"), dur=s.get("dur"), framing=s.get("framing"), who=s.get("who"),
                  motion=s.get("motion_prompt"), lines=s.get("lines"), seconds=round(n / fps, 2), v=2)
json.dump(idx, open(idx_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(idx), "sheets in", a.out)
