"""Save a finished judge workflow's result (the task output file) as judge_<tag>.json and print the keep counts.
  python train/save_judge.py <task output file> train/pilot/<char>/judge_w1.json
"""
import json, sys, collections
src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8").read()
data = json.loads(t)["result"]
json.dump(data, open(dst, "w", encoding="utf-8"), indent=1)
ims = [im for g in data for im in g["images"]]
print(len(ims), "images", dict(collections.Counter(im["keep"] for im in ims)), dict(collections.Counter(im["style"] for im in ims)))
for im in ims:
    if im["keep"] != "no": print(im["keep"], im["file"], im["actual_shot"], im["style"][:10], "|", im["flaws"][:110])
