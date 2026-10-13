"""SeedVR2 post pass for approved clips (2026-10-13; the user approved SeedVR2 after the side-by-side flicker check).
Reads shots/ep01_post/manifest.json (tools/post_manifest.py), trims each clip to its usable span (verdict trim: from-to; pass: 0 to
min(clip, dur + 0.5 s)), upscales it with SeedVR2 3B (the settings measured in train/seedvr2_test.py: int8 model, decode tiles 256 px /
16 frames, lab colour correction, denoise 1.0) and writes shots/ep01_post/<id>/final_1080p.mp4 (16 fps, frame interpolation comes later)
plus shots/ep01_post/times.csv. Skips shots whose final was made from the same clip and span. Run through train/comfy_run.py:
  python train/comfy_run.py tools/seedvr2_post.py [--only b01_02,b01_05]
"""
import argparse, csv, json, os, sys, time
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE); sys.path.insert(0, HERE)
import run_shots as R  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "train"))
HOST = "127.0.0.1:8188"; P = "shots/ep01_post"


def wf(uploaded, prefix):
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": uploaded}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "upscale_method": "lanczos", "width": 1920, "height": 1088, "crop": "disabled"}},
        "4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "seedvr2_ema_vae_fp16.safetensors"}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": "seedvr2_3b_int8_convrot.safetensors", "weight_dtype": "default"}},
        "7": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["4", 0], "vae": ["5", 0], "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}},
        "8": {"class_type": "SeedVR2TemporalChunk", "inputs": {"latent": ["7", 0], "temporal_overlap": 2, "chunking_mode": "auto"}},
        "9": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["6", 0], "vae_conditioning": ["8", 0]}},
        "10": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["8", 0], "seed": 42,
                                                    "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["10", 0], "temporal_overlap": ["8", 1]}},
        "12": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["11", 0], "vae": ["5", 0], "tile_size": 256, "overlap": 64, "temporal_size": 16, "temporal_overlap": 4}},
        "13": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["12", 0], "original_resized_images": ["3", 0], "color_correction_method": "lab"}},
        "14": {"class_type": "ImageScale", "inputs": {"image": ["13", 0], "upscale_method": "lanczos", "width": 1920, "height": 1080, "crop": "center"}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": ["2", 2]}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}},
    }


def trim(src, dst, t0, t1):
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 16; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a, b = max(0, int(round(t0 * fps))), min(n, int(round(t1 * fps)) + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a); w = None; k = 0
    for _ in range(a, b):
        ok, f = cap.read()
        if not ok:
            break
        if w is None:
            w = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
        w.write(f); k += 1
    w.release(); return k, fps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); a = ap.parse_args()
    man = json.load(open(f"{P}/manifest.json", encoding="utf-8"))
    only = set(a.only.split(",")) if a.only else None
    tf = open(f"{P}/times.csv", "a", newline="", encoding="utf-8"); tw = csv.writer(tf)
    for k, m in man.items():
        if only and k not in only:
            continue
        d = f"{P}/{k}"; os.makedirs(d, exist_ok=True); meta = f"{d}/final.json"
        t0 = float(m["from"]); t1 = float(m["to"]) if m["verdict"] == "trim" else min(float(m["to"]), float(m["dur"] or 5) + 0.5)
        key = {"clip": m["clip"], "from": t0, "to": t1}
        if os.path.exists(f"{d}/final_1080p.mp4") and os.path.exists(meta) and json.load(open(meta, encoding="utf-8")).get("key") == key:
            continue
        src = f"{d}/trimmed.mp4"; n, fps = trim(m["clip"], src, t0, t1)
        if n < 8:
            print(f"[{k}] skipped: only {n} frames in span", flush=True); continue
        st = time.time()
        up = R.upload_image(HOST, src, name=f"post_{k}.mp4")
        outs, _ = R.wait(HOST, R.submit(HOST, wf(up, f"GameStories/post/{k}")), poll=5)
        for no in outs.values():
            for v in no.get("images", []) + no.get("video", []) + no.get("gifs", []):
                R.download(HOST, v, f"{d}/final_1080p.mp4")
        dt = time.time() - st
        json.dump({"key": key, "frames": n, "fps": fps, "seconds": round(dt, 1)}, open(meta, "w", encoding="utf-8"), indent=1)
        tw.writerow([k, m["clip"], t0, t1, n, round(dt, 1)]); tf.flush()
        print(f"[{k}] {n} frames {t0:.1f}-{t1:.1f}s -> final_1080p.mp4 in {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
