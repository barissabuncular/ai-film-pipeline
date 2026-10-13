"""Masked inpaint with Z-Image Turbo + an optional character LoRA (2026-10-12): repaint only a masked region of a keyframe, e.g. replace a
Butcher-LoRA-bled warrior with Aidan (aidnw LoRA) while the Butcher's pixels outside the mask stay. Workflow built in code (VAEEncode ->
SetLatentNoiseMask -> KSampler at the row's denoise); the result is composited back over the original through a feathered copy of the mask,
so nothing outside the mask changes. Run through train/comfy_run.py (it starts and stops ComfyUI):
  python train/comfy_run.py tools/run_inpaint.py --csv <rows.csv> --out-dir shots/ep01_v3_edits
CSV columns: id, input, mask (white = repaint), prompt, lora (file or none), lora_strength, denoise, seed, count.
Output: <out-dir>/<id>/ip_<seed>.png (composited, same size as the input) and ip_<seed>_raw.png.
"""
import argparse, csv, json, os, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from run_shots import submit, wait, download, upload_image  # noqa: E402


def build(img, mask, prompt, lora, strength, denoise, seed, prefix):
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "z_image_ae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "20": {"class_type": "LoadImage", "inputs": {"image": img}},
        "21": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}},
        "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["3", 0]}},
        "23": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["22", 0], "mask": ["21", 0]}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["23", 0], "seed": seed,
                                                    "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }
    if lora and lora.lower() != "none":
        wf["11"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": float(strength or 1.0)}}
        wf["4"]["inputs"]["model"] = ["11", 0]
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--host", default="127.0.0.1:8188")
    a = ap.parse_args()
    for r in csv.DictReader(open(os.path.join(HERE, a.csv), encoding="utf-8")):
        d = os.path.join(HERE, a.out_dir, r["id"]); os.makedirs(d, exist_ok=True)
        img = upload_image(a.host, r["input"], name=f"ip_{r['id']}_in.png")
        msk = upload_image(a.host, r["mask"], name=f"ip_{r['id']}_mask.png")
        orig = cv2.imread(r["input"]); m = cv2.imread(r["mask"], 0).astype(np.float32) / 255
        m = cv2.GaussianBlur(m, (31, 31), 0)[..., None]
        for i in range(int(r.get("count") or 1)):
            seed = int(r["seed"]) + i; dest = os.path.join(d, f"ip_{seed}.png")
            if os.path.exists(dest):
                continue
            pid = submit(a.host, build(img, msk, r["prompt"], r.get("lora"), r.get("lora_strength"), float(r["denoise"]), seed, f"GameStories/inpaint/{r['id']}"))
            outs, dt = wait(a.host, pid, poll=3)
            for no in outs.values():
                for im in no.get("images", []):
                    raw = dest[:-4] + "_raw.png"; download(a.host, im, raw)
                    gen = cv2.resize(cv2.imread(raw), (orig.shape[1], orig.shape[0]))
                    cv2.imwrite(dest, (orig * (1 - m) + gen * m).round().astype(np.uint8))
                    print(f"[{r['id']}] seed={seed} denoise={r['denoise']} {dt:.0f}s -> {os.path.relpath(dest, HERE)}", flush=True)


if __name__ == "__main__":
    main()
