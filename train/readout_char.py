"""Pass/fail readout for a character LoRA without an earlier version (CPU), after night_char.py and train/judge_eval_char.js.

Rule (fixed before the eval renders; for the Butcher on 2026-10-05):
- Main columns (--main) are compared with the no-LoRA baseline on the same scene/seed pairs, judged blind in one pass.
- A main column PASSES when all hold:
  1. on model (judge identity 6+ and costume 7+) in 70% of its images
  2. it beats the baseline: exact McNemar, one-sided p < 0.05 / number of main columns
  3. mean style_photoreal 7 or more (the edit model's game-art look must not have been learned)
  4. mean artifacts no more than 1.0 below the baseline's, and extra objects in at most 10% of its images
- Other columns are exploratory. The face score (AuraFace vs the anchors) is printed when available, not used.
  python train/readout_char.py --name bchr_v1 --judge train/pilot/butcher/eval_bchr_v1_judge.json --main c1500,c2000,c2500
"""
import argparse, collections, csv, glob, json, math, os

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--judge", required=True); ap.add_argument("--main", required=True)
a = ap.parse_args()
D = f"train/pilot/eval_{a.name}"; B = f"train/pilot/eval_{a.name}_blind"
key = json.load(open(f"logs/eval_{a.name}_blind_key.json"))
given = {}
for f in glob.glob(f"{B}/prompts_*.json"):
    for k in json.load(open(f, encoding="utf-8")):
        d, img = k.split("/"); scn, x = d.rsplit("_", 1)
        given[(key[x], scn, int(img[3:-4]))] = True
jd = json.load(open(a.judge, encoding="utf-8"))
if jd.get("groups_missing") or not jd.get("judged"):
    raise SystemExit(f"judge pass incomplete: groups_missing={jd.get('groups_missing')}")
J = {}
for g in jd["judged"]:
    for r in g["rows"]:
        x = str(r["column"]).strip().rsplit("_", 1)[-1].upper()
        J[(key.get(x, x), str(r["scene"]).strip(), int(r["seed"]))] = r
unknown = sorted(set(J) - set(given)); lost = collections.Counter(k[0] for k in set(given) - set(J))
if unknown or lost:
    raise SystemExit(f"judge rows do not match the images given: unknown {unknown[:3]}, skipped {dict(lost)}")
sc = {}
if os.path.exists(f"{D}/face_scores.csv"):
    sc = {r["file"].replace("\\", "/"): r for r in csv.DictReader(open(f"{D}/face_scores.csv", encoding="utf-8"))}
on = lambda r: r["identity"] >= 6 and r["costume"] >= 7
cols = sorted({k[0] for k in J}, key=lambda c: (c != "base", c))


def stats(c):
    rs = {k[1:]: v for k, v in J.items() if k[0] == c}
    m = lambda f: sum(r[f] for r in rs.values()) / max(1, len(rs))
    fs = [float(sc[f"{D}/{s}_{c}/kf_{sd}.png"]["aura"]) for (s, sd) in rs if sc.get(f"{D}/{s}_{c}/kf_{sd}.png", {}).get("aura")]
    return dict(n=len(rs), on=sum(on(r) for r in rs.values()), id=m("identity"), cos=m("costume"), pf=m("prompt_following"), bl=m("background_bleed"),
                art=m("artifacts"), sty=m("style_photoreal"), extra=sum(r["extra_object"] for r in rs.values()),
                face=sum(fs) / len(fs) if fs else float("nan"), rows=rs)


def mcnemar(x, y):
    ka = sum(on(r) and not on(y["rows"][k]) for k, r in x["rows"].items() if k in y["rows"])
    kb = sum(on(y["rows"][k]) and not on(r) for k, r in x["rows"].items() if k in y["rows"])
    n = ka + kb
    return ka, kb, (sum(math.comb(n, i) for i in range(ka, n + 1)) / 2 ** n if n else 1.0)


S = {c: stats(c) for c in cols}
print(f"{'column':7s} {'on model':>9s}  identity costume prompt bleed artifacts photoreal extra  face")
for c in cols:
    s = S[c]
    print(f"{c:7s} {s['on']:3d}/{s['n']:2d}     {s['id']:5.1f} {s['cos']:7.1f} {s['pf']:6.1f} {s['bl']:5.1f} {s['art']:7.1f} {s['sty']:9.1f} {s['extra']:5d}  {s['face']:.3f}")
main = [c for c in a.main.split(",") if c in S]; alpha = 0.05 / max(1, len(main)); res = []
print(f"\nMain columns vs the no-LoRA baseline (p < {alpha:.3f}):")
for c in main:
    s, b = S[c], S["base"]; ka, kb, p = mcnemar(s, b)
    checks = {"on model 70%": s["on"] >= 0.7 * s["n"], f"beats baseline": p < alpha, "photoreal 7+": s["sty"] >= 7,
              "artifacts held": s["art"] >= b["art"] - 1.0 and s["extra"] <= 0.1 * s["n"]}
    ok = all(checks.values()); res.append((c, ok))
    print(f"  {c}: on model {s['on']} vs {b['on']} (only LoRA {ka}, only baseline {kb}, p={p:.4f}); "
          + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in checks.items()) + f"  => {'PASS' if ok else 'fail'}")
print("\nRESULT: " + ("PASS with " + ", ".join(c for c, ok in res if ok) if any(ok for _, ok in res) else "no main column passes"))
