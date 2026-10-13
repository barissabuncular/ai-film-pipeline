"""Print the train/judge_cands_char.js args (JSON) for one character's edit pass, from train/pilot/judge_specs_2026-10-06.json.
Folders of anchor-fix rows (meta "char") are skipped; groups of 4 folders.
  python train/make_judge_args.py lazarus [w1]
"""
import json, os, sys
ch, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "w1")
d = f"train/pilot/{ch}/{tag}"
M = json.load(open(f"train/pilot/{ch}/{tag}_meta.json", encoding="utf-8"))
f = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) and any(p.endswith(".png") for p in os.listdir(os.path.join(d, x))) and not M.get(x, {}).get("char"))
groups = [f[i:i + 4] for i in range(0, len(f), 4)]
S = json.load(open("train/pilot/judge_specs_2026-10-06.json", encoding="utf-8")); s = S[ch]; G = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/") + "/"
print(json.dumps({"dir": G + d, "meta": G + f"train/pilot/{ch}/{tag}_meta.json", "groups": groups, "who": s["who"], "refs": [G + r for r in s["refs"]],
                  "checklist": s["checklist"], "rules": s["rules"] + " " + S["_dup_rule"] + " " + S["_style_rule"]}))
