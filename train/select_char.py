"""Select a character's training set from judged edit passes and write the dataset (CPU). Generalised from
train/pilot/butcher/select_butcher.py. Config (JSON): trigger, dir, tags (edit passes, each with <tag>_meta.json and judge_<tag>.json),
out, targets {shot: n}, places {key: caption phrase}, lights {key: phrase}, default_place/default_light (pass-1 images with no place,
e.g. the studio anchor), fixed (regex of identity/kit words never captioned), near_res, far_res.
Rules: one full image per figure (a background swap keeps the figure pixel-identical) plus at most one judged waist-up crop;
at most 3 images per place (4 for the default/studio place, +1 for crops); "maybe" images only through their crop box;
style must be photoreal or slightly glossy (cartoon never). Captions: trigger, shot, view, action, place, light.
  python train/select_char.py train/pilot/moreina/select.json
"""
import collections, csv, json, os, re, shutil, sys
from PIL import Image

C = json.load(open(sys.argv[1], encoding="utf-8"))
B, OUT, TRIG = C["dir"], C["out"], C["trigger"]
FIXED = re.compile(C["fixed"], re.I)
TARGET = C["targets"]


def load(tag):
    meta = json.load(open(f"{B}/{tag}_meta.json", encoding="utf-8"))
    out = []
    for g in json.load(open(f"{B}/judge_{tag}.json", encoding="utf-8")):
        for im in g["images"]:
            folder = im["file"].split("/")[0]; m = meta.get(folder, {})
            # photoreal_only: drop "slightly glossy" images too. (Moreina v1 came out glossier than the baseline, 6.6 vs 8.1, although every
            # selected image was judged photoreal: the edit model's airbrushed skin adds up in the LoRA. The Butcher, whose images all went
            # through the Z-Image photoreal pass, scored 7.4-7.7 vs 6.8. So a texture pass is the likely fix, not this filter alone.)
            bad_style = not im.get("style", "").startswith("photoreal") if C.get("photoreal_only") else im.get("style", "").startswith("cartoon")
            if m.get("char") or im["keep"] == "no" or (im["keep"] == "maybe" and not im.get("waist_up_box")) \
                    or bad_style or (m.get("place") and im.get("anchor_background")):
                continue
            q = (im["same_person"] == "yes") * 2 + (im["kit_ok"] == "yes") + (im.get("place_ok") in ("yes", "not asked")) \
                + (im.get("light_ok") in ("yes", "not asked")) + 2 * im.get("style", "").startswith("photoreal")
            fig = m.get("fig") or im["file"][:-4]
            out.append(dict(file=f"{B}/{tag}/{im['file']}", fig=fig, tag=tag, q=q, maybe=im["keep"] == "maybe", shot=im["actual_shot"], view=im["actual_view"],
                            place=m.get("place", C["default_place"]), light=m.get("light", C["default_light"]), states=im["states"], box=im.get("waist_up_box", "")))
    return out


def action(states):
    parts = [p.strip() for p in re.split(r"[;,]", states) if p.strip() and not FIXED.search(p)]
    return ", ".join(p for p in parts[:2] if len(p.split()) <= 9)


cands = sum((load(t) for t in C["tags"] if os.path.exists(f"{B}/judge_{t}.json")), [])
cands.sort(key=lambda c: (-c["q"], c["tag"] == C["tags"][0]))
used_fig = collections.Counter(); used_place = collections.Counter(); bins = collections.Counter(); chosen = []
cap = lambda p: 4 if p == C["default_place"] else 3


def take(c, shot, crop=None):
    chosen.append(dict(c, shot=shot, crop=crop)); bins[shot] += 1; used_place[c["place"]] += 1; used_fig[c["fig"]] += 1


for c in cands:
    if not c["maybe"] and bins[c["shot"]] < TARGET.get(c["shot"], 0) and not used_fig[c["fig"]] and used_place[c["place"]] < cap(c["place"]):
        take(c, c["shot"])
for prefer_new in (True, False):
    for c in cands:
        if bins["medium shot"] >= TARGET.get("medium shot", 0):
            break
        box = [int(v) for v in re.findall(r"\d+", c["box"] or "")]
        used_file = c["file"] in {x["file"] for x in chosen}
        if c["shot"] == "full body" and len(box) == 4 and used_fig[c["fig"]] <= 1 and not (prefer_new and used_file) \
                and not any(x["file"] == c["file"] and x["crop"] for x in chosen) \
                and used_place[c["place"]] < cap(c["place"]) + 1 and box[2] - box[0] >= 300 and box[3] - box[1] >= 300:
            take(c, "medium shot", box)
print(f"{len(chosen)} selected from {len(cands)} candidates; bins {dict(bins)}; places {dict(used_place)}")
shutil.rmtree(OUT, ignore_errors=True)
for d in ("near", "far"):
    os.makedirs(f"{OUT}/{d}")
rows = []
for i, c in enumerate(chosen):
    d = "far" if c["shot"] == "full body" else "near"
    im = Image.open(c["file"]).convert("RGB")
    if c["crop"]:
        x0, y0, x1, y1 = c["crop"]; im = im.crop((max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1)))
    name = f"{i:02d}_{os.path.basename(os.path.dirname(c['file']))[:40]}{'_crop' if c['crop'] else ''}"
    im.save(f"{OUT}/{d}/{name}.png")
    cap_ = ", ".join(p for p in [TRIG, c["shot"], c["view"], action(c["states"]), C["places"].get(c["place"], ""), C["lights"].get(c["light"], "")] if p)
    open(f"{OUT}/{d}/{name}.txt", "w", encoding="utf-8").write(cap_)
    rows.append(dict(file=c["file"], out=f"{d}/{name}.png", crop=c["crop"] or "", fig=c["fig"], caption=cap_))
with open(f"{OUT}/selection.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
G = os.path.abspath(".").replace("\\", "/")
open(f"{OUT}/dataset.toml", "w").write(f"""# {TRIG} LoRA dataset, built by train/select_char.py {sys.argv[1]}
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
resolution = [{C['near_res']}, {C['near_res']}]
image_directory = "{G}/{OUT}/near"
cache_directory = "{G}/{OUT}/cache_near"
num_repeats = 1

[[datasets]]
resolution = [{C['far_res']}, {C['far_res']}]
image_directory = "{G}/{OUT}/far"
cache_directory = "{G}/{OUT}/cache_far"
num_repeats = 1
""")
print("dataset ->", OUT)
