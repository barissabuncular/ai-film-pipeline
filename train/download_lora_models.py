"""Download the models for the character-LoRA pipeline (story/diablo/episodes/ep01_the_butcher/lora_plan.md).

All Apache 2.0 or MIT, none gated (checked 2026-09-30). Every file is verified against the SHA256 that Hugging Face
publishes for it; a mismatch deletes the file and fails. Already-present files with the right hash are skipped.

Run with the ComfyUI Python:
  python train\\download_lora_models.py [name-filter ...]
A filter keeps only files whose name contains one of the given strings (skips re-hashing the rest).
"""
import hashlib, os, shutil, sys, time
from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "ComfyUI_windows_portable"))
M = rf"{COMFY_ROOT}\ComfyUI\models"
T = rf"{PROJECT_ROOT}\tools\models"

# (repo, file in repo, destination folder, destination name or None, expected sha256 prefix or None)
FILES = [
    ("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf", rf"{M}\unet", None, "8677bac90627"),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", rf"{M}\text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "cb5636d852a0"),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", rf"{M}\vae", "qwen_image_vae.safetensors", "a70580f0213e"),
    ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", rf"{M}\loras", None, "22226e8d05d3"),
    ("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "qwen-image-edit-2511-multiple-angles-lora.safetensors", rf"{M}\loras", None, "42426ded4e25"),
    ("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "comfyui-workflow-multiple-angles.json", rf"{PROJECT_ROOT}\workflows\reference", None, None),
    ("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors", rf"{M}\diffusion_models", None, "97ed34fe0567"),
    ("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "split_files/vae/flux2-vae.safetensors", rf"{M}\vae", "flux2-vae.safetensors", "868fe7b343cc"),
    ("fal/AuraFace-v1", "glintr100.onnx", rf"{T}\auraface", None, "a7933ea53301"),
    ("facebook/dinov2-base", "config.json", rf"{T}\dinov2-base", None, None),
    ("facebook/dinov2-base", "preprocessor_config.json", rf"{T}\dinov2-base", None, None),
    ("facebook/dinov2-base", "model.safetensors", rf"{T}\dinov2-base", None, "d73036b56966"),
    ("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx", rf"{T}\opencv", None, "8f2383e4dd3c"),
    ("opencv/face_recognition_sface", "face_recognition_sface_2021dec.onnx", rf"{T}\opencv", None, "0ba9fbfa01b5"),
    # Route B (lora_plan.md): de-distilled Turbo for adapter-free LoRA training, Apache 2.0 (checked 2026-10-03)
    ("ostris/Z-Image-De-Turbo", "z_image_de_turbo_v1_bf16.safetensors", rf"{PROJECT_ROOT}\train\models", None, "c57d8e5b1363"),
    ("ostris/Z-Image-De-Turbo", "README.md", rf"{PROJECT_ROOT}\train\licences\ostris_Z-Image-De-Turbo", None, None),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    api = HfApi()
    failed = []
    only = sys.argv[1:]
    for repo, fname, dest_dir, dest_name, pin in FILES:
        if only and not any(o in fname or o in repo for o in only):
            continue
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, dest_name or os.path.basename(fname))
        info = api.model_info(repo, files_metadata=True)
        sib = next(s for s in info.siblings if s.rfilename == fname)
        want = sib.lfs.sha256 if sib.lfs else None
        if pin and want and not want.startswith(pin):
            print(f"[FAIL] {fname}: published sha {want[:12]} != pinned {pin}", flush=True)
            failed.append(fname)
            continue
        if os.path.exists(dest) and (want is None or sha256(dest) == want):
            print(f"[skip] {dest} already present and verified", flush=True)
            continue
        t0 = time.time()
        print(f"[get ] {repo}/{fname} ({round((sib.size or 0) / 1e9, 2)} GB) -> {dest}", flush=True)
        tmp_dir = os.path.join(dest_dir, ".dl_tmp")
        p = hf_hub_download(repo, fname, local_dir=tmp_dir)
        if want:
            got = sha256(p)
            if got != want:
                os.remove(p)
                print(f"[FAIL] {fname}: sha {got[:12]} != {want[:12]}, deleted", flush=True)
                failed.append(fname)
                continue
        shutil.move(p, dest)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        dt = time.time() - t0
        print(f"[ ok ] {dest}  {round(os.path.getsize(dest) / 1e9, 2)} GB in {dt / 60:.1f} min, sha verified={bool(want)}", flush=True)
    print("DONE" if not failed else f"DONE with failures: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
