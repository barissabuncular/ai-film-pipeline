"""Build a character LoRA dataset from selected edit outputs (plan step 7, story/diablo/episodes/ep01_the_butcher/lora_plan.md).

Input: a selection CSV with columns file, keep (yes/no) and optional caption_extra. Captions are written from the edit
row id (tools/run_edit.py folders): trigger first, then only what VARIES (view, distance, expression, light, place).
Fixed face and costume words are never captioned, so the LoRA learns them as part of the trigger.
Close and medium shots go to near/ (trained at --near-res), full-body shots to far/ (--far-res).
Writes dataset.toml (two datasets), PROVENANCE.csv and a caption lint report.

  python train/build_lora_dataset.py --select train/pilot/aidan/selection.csv --trigger "aidnw man" --out train/datasets/aidnw
"""
import argparse, csv, os, re, shutil

VIEWS = ["front view", "three-quarter view from his right", "right side profile", "back three-quarter view",
         "back view", "back three-quarter view", "left side profile", "three-quarter view from his left"]
SHOT = {"close": "close-up", "medium": "medium shot", "low": "low-angle medium shot", "high": "high-angle medium shot"}
EXPR = {"angry": "shouting in anger", "grief": "in grief, eyes wet", "fear": "afraid, eyes wide", "exhausted": "exhausted, eyes half closed",
        "resolve": "grim determined expression", "speaking": "speaking, mouth open", "faintsmile": "faint tired smile", "eyesclosed": "eyes closed, head bowed"}
TURN = {"lookleft": "looking to his left", "lookright": "looking to his right", "lookup": "looking up", "lookdown": "looking down"}
LIGHT = {"torchnight": "at night, lit by torchlight", "overcast": "overcast daylight, drizzle", "dusk": "against a crimson dusk sky",
         "candle": "inside a candle-lit tavern", "crypt": "in a dark crypt, firelight from below", "rain": "at night in heavy rain"}
PLACE = {"tavern": "inside a smoky tavern", "grave": "kneeling at a grave on a hill", "nave": "in a ruined church nave",
         "catacomb": "holding a torch in a catacomb", "fire": "sitting by a campfire at night", "drawsword": "drawing his sword",
         "shieldup": "raising his shield, sword in hand", "road": "walking on a muddy road in the rain"}
FULL = {"front": "full body, standing, front view", "walking": "full body, walking toward the camera", "back": "full body, back view, walking away",
        "side": "full body, side view", "nave": "full body, standing in a church nave", "doorway": "full body, standing in a tavern doorway",
        "fightstance": "full body, fighting stance with sword and shield", "resting": "full body, leaning on his sword"}
# words that describe the fixed look; actions with a prop ("raising his shield") are fine, its colour or make is not
FIXED_WORDS = re.compile(r"\b(surcoat|mail|pauldrons?|cloak|beard|hair|scar|grey eyes|dark red|round shield)\b", re.I)


def caption(row_id, extra=""):
    m = re.match(r"b_az(\d)_(\w+)", row_id)
    if m:
        parts = [VIEWS[int(m.group(1))], SHOT[m.group(2)], "in a muddy village street at dusk"]
        far = False
    elif row_id.startswith("b_full_az"):
        parts = ["full body", VIEWS[int(row_id[-1])], "in a muddy village street at dusk"]; far = True
    elif row_id.startswith("c_expr_"):
        parts = ["close-up", EXPR[row_id[7:]]]; far = False
    elif row_id.startswith("c_turn_"):
        parts = ["close-up", TURN[row_id[7:]]]; far = False
    elif row_id.startswith("d_light_"):
        parts = ["medium shot", LIGHT[row_id[8:]]]; far = False
    elif row_id.startswith("e_place_"):
        parts = ["medium shot", PLACE[row_id[8:]]]; far = False
    elif row_id.startswith("f_full_"):
        parts = [FULL[row_id[7:]]]; far = True
    elif row_id in ("A0", "anchor"):
        parts = ["medium shot", "front view", "in a muddy village street at dusk, torch on a pole"]; far = False
    else:
        raise ValueError(f"no caption rule for {row_id}")
    if extra:
        parts.append(extra)
    return ", ".join(parts), far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--near-res", type=int, default=768)
    ap.add_argument("--far-res", type=int, default=1024)
    a = ap.parse_args()
    for sub in ("near", "far"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    prov = []; lint = []
    for r in csv.DictReader(open(a.select, encoding="utf-8")):
        if r.get("keep", "yes").strip().lower() != "yes":
            continue
        src = r["file"].replace("\\", "/")
        row_id = r.get("row_id") or src.split("/")[-2]
        if r.get("caption"):  # a reviewed caption (what the image actually shows) wins over the row-id rule
            cap, far = r["caption"], r.get("far", "no").strip().lower() == "yes"
        else:
            cap, far = caption(row_id, r.get("caption_extra", ""))
        cap = f"{a.trigger}, {cap}"
        if FIXED_WORDS.search(cap.split(",", 1)[1]):
            lint.append(f"{src}: fixed identity/costume word in caption: {cap}")
        sub = "far" if far else "near"
        name = f"{row_id}_{os.path.splitext(os.path.basename(src))[0]}"
        shutil.copy2(src, os.path.join(a.out, sub, name + ".png"))
        open(os.path.join(a.out, sub, name + ".txt"), "w", encoding="utf-8").write(cap)
        prov.append({"image": f"{sub}/{name}.png", "source": src, "row_id": row_id, "caption": cap})
    with open(os.path.join(a.out, "PROVENANCE.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "source", "row_id", "caption"]); w.writeheader(); w.writerows(prov)
    out = os.path.abspath(a.out).replace("\\", "/")
    toml = (f'# {a.trigger} LoRA dataset, built by train/build_lora_dataset.py\n[general]\ncaption_extension = ".txt"\nbatch_size = 1\n'
            f'enable_bucket = true\nbucket_no_upscale = true\n\n'
            f'[[datasets]]\nresolution = [{a.near_res}, {a.near_res}]\nimage_directory = "{out}/near"\ncache_directory = "{out}/cache_near"\nnum_repeats = 1\n\n'
            f'[[datasets]]\nresolution = [{a.far_res}, {a.far_res}]\nimage_directory = "{out}/far"\ncache_directory = "{out}/cache_far"\nnum_repeats = 1\n')
    open(os.path.join(a.out, "dataset.toml"), "w", encoding="utf-8").write(toml)
    n_near = sum(1 for p in prov if p["image"].startswith("near")); n_far = len(prov) - n_near
    print(f"{len(prov)} images ({n_near} near @ {a.near_res}, {n_far} far @ {a.far_res}) -> {a.out}")
    for l in lint:
        print("LINT", l)


if __name__ == "__main__":
    main()
