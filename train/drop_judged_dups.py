"""Set to 'no' every kept image whose judge flaws say 'near duplicate of ed_NNNNNN' while that other image is kept (the judge names the
weaker one of a pair). Usage: python train/drop_judged_dups.py train/pilot/<char>/judge_w2.json"""
import json, re, sys
p = sys.argv[1]; d = json.load(open(p, encoding="utf-8"))
keep = {im["file"].split("/")[-1]: im for g in d for im in g["images"] if im["keep"] != "no"}
n = 0
for im in list(keep.values()):
    m = re.search(r"near[- ]duplicate of (ed_\d+)(?:\.png)?", im["flaws"], re.I)  # judges write it with or without .png
    if m and m.group(1) + ".png" in keep and m.group(1) + ".png" != im["file"].split("/")[-1] and keep[m.group(1) + ".png"]["keep"] != "no":
        im["keep"] = "no"; im["flaws"] = f"SET TO NO (lead, auto): near duplicate of {m.group(1)}, which is kept. " + im["flaws"]; n += 1
json.dump(d, open(p, "w", encoding="utf-8"), indent=1); print(n, "near duplicates set to no")
