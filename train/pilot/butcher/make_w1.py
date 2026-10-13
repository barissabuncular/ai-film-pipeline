"""Butcher LoRA, edit pass 1 (anchor pack + poses) from the approved concept A0 (r6/C17_7/kf_40402.png).
Plan (lora_plan.md, Butcher): 30 training images = full body 12, medium 10 (4 cleaver raised), head 5, charging 3; blood is a
captioned state, at least 40% 'old blood on the apron, clean red skin'; no corpses, wounds or body parts in any image.
Angle rows (multiple-angles LoRA) first, then text edits, so the LoRA is switched once. Writes w1.csv and w1_meta.json.
  python train/pilot/butcher/make_w1.py ; python tools/run_edit.py --csv train/pilot/butcher/w1.csv --out-dir train/pilot/butcher/w1
"""
import csv, json

B = "train/pilot/butcher"
KEEP = ("Keep this same demon exactly the same: his huge obese red-skinned body and sagging belly, his head with two curved cream horns, "
        "pointed ears, small fierce eyes and tusked fanged mouth, the pale canvas apron with shoulder straps, his clawed hands and feet, "
        "and the huge rusted cleaver")
rows, meta, seed = [], {}, [200000]


def add(id, src, prompt, shot, view, angles=0.0, n=2, **m):
    rows.append({"id": id, "input": src, "input2": "", "prompt": prompt, "angles": angles, "seed": seed[0], "count": n}); seed[0] += 10
    meta[id] = dict(shot=shot, view=view, source=src, **m)


# camera turns with the multiple-angles LoRA (Aidan: reliable around 8 azimuths, left/right swapped in its naming)
for i, (v, lab) in enumerate([("front view", "front view"), ("front-right quarter view", "three-quarter view"), ("right side view", "side view"),
                              ("back-right quarter view", "back three-quarter view"), ("back view", "back view"), ("left side view", "side view"),
                              ("front-left quarter view", "three-quarter view")]):
    add(f"az{i}_medium", f"{B}/A0.png", f"<sks> {v} eye-level shot medium shot", "medium shot", lab, angles=1.0, n=1)
    add(f"az{i}_full", f"{B}/A0_portrait.png", f"<sks> {v} eye-level shot wide shot", "full body", lab, angles=1.0, n=1)
add("az0_low", f"{B}/A0.png", "<sks> front view low-angle shot medium shot", "medium shot", "low-angle view from the front", angles=1.0, n=2)
# blood state: clean skin, old blood on the apron only (thumbnail-safe tier)
add("clean_A0", f"{B}/A0.png", "Remove all the fresh blood from his skin, arms, belly and legs so his red skin is clean. Keep only a few old dark "
    "brown blood stains splattered on his pale apron. Keep his face, horns, pose, cleaver, the doorway, the light and the camera exactly the same.",
    "medium shot", "front view", n=3, blood="old")
add("clean_portrait", f"{B}/A0_portrait.png", "Remove all the fresh blood from his skin, arms, belly and legs so his red skin is clean. Keep only "
    "a few old dark brown blood stains splattered on his pale apron. Keep his face, horns, pose, cleaver, the background, the light and the camera "
    "exactly the same.", "full body", "front view", n=2, blood="old")
# head close-ups (A1 candidates)
add("head_front", f"{B}/A0.png", f"Close-up portrait of this same demon, his horned head and huge shoulders filling the frame, facing the camera. {KEEP}.",
    "close-up", "front view", n=4)
add("head_roar", f"{B}/A0.png", f"Close-up portrait of this same demon roaring, mouth wide open showing his tusks and fangs, his horned head and shoulders "
    f"filling the frame. {KEEP}.", "close-up", "front view", n=2, expr="roaring")
add("head_34", f"{B}/A0.png", f"Close-up portrait of this same demon in three-quarter view, his horned head and shoulders filling the frame, glaring to the "
    f"side. {KEEP}.", "close-up", "three-quarter view", n=2)
add("head_low", f"{B}/A0.png", f"Low-angle close-up looking up at this same demon's horned head and huge shoulders, leering down at the camera. {KEEP}.",
    "close-up", "low-angle view from the front", n=2)
# poses
for id, act, shot, n in [("pose_raised2h", "raising the huge cleaver high over his head with both hands, about to strike", "medium shot", 2),
                         ("pose_swing", "swinging the huge cleaver sideways in a wide arc", "medium shot", 2),
                         ("pose_stand", "standing still and hunched, the huge cleaver hanging at his side, glaring at the camera", "full body", 2),
                         ("pose_stoop", "stooping under a low timber beam, head bowed, the cleaver in his fist", "medium shot", 2),
                         ("pose_charge", "charging straight at the camera with heavy running steps, the cleaver raised", "full body", 2),
                         ("pose_back", "seen from behind, walking away into darkness, the cleaver hanging at his side, his horns and the apron straps across his "
                          "broad red back visible", "full body", 2),
                         ("pose_roar", "standing with his head thrown back, roaring, the cleaver raised in one fist", "full body", 2)]:
    add(id, f"{B}/A0_portrait.png", f"Show this same demon {act}, his whole body visible from horns to clawed feet. {KEEP}." if shot == "full body"
        else f"Show this same demon from the waist up, {act}. {KEEP}.", shot, "back view" if "back" in id else "front view", n=n, act=act)

with open(f"{B}/w1.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "input", "input2", "prompt", "angles", "seed", "count"]); w.writeheader(); w.writerows(rows)
json.dump(meta, open(f"{B}/w1_meta.json", "w", encoding="utf-8"), indent=1)
print(len(rows), "rows", sum(r["count"] for r in rows), "edits")
