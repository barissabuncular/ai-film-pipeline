"""Convert musubi LoRA checkpoints for ComfyUI by hand (2026-10-11: a training stopped before its end-of-run conversion).
  python train/convert_ckpt.py <musubi ckpt> <comfy name without .safetensors> [<ckpt> <name> ...]
"""
import os, subprocess, sys
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "ComfyUI_windows_portable"))
MUSUBI_ROOT = os.environ.get("MUSUBI_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "musubi-tuner"))
MT = MUSUBI_ROOT; MPY = rf"{MT}\.venv\Scripts\python.exe"; LORAS = rf"{COMFY_ROOT}\ComfyUI\models\loras"
a = sys.argv[1:]
for ck, name in zip(a[::2], a[1::2]):
    dest = os.path.join(LORAS, name + ".safetensors")
    if os.path.exists(dest):
        print("exists", dest); continue
    r = subprocess.run([MPY, r"src\musubi_tuner\convert_lora.py", "--input", os.path.abspath(ck), "--output", dest, "--target", "other"], cwd=MT)
    print("converted" if r.returncode == 0 and os.path.exists(dest) else "FAILED", ck, "->", dest, flush=True)
