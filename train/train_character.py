"""Train one character LoRA for Z-Image Turbo with musubi-tuner (plan step 8), then convert every checkpoint
for ComfyUI. ComfyUI must be stopped. Settings follow the measured timing test (train/timing/results.csv).

  python train/train_character.py --name aidnw --dataset train/datasets/aidnw/dataset.toml      (the ComfyUI portable Python)
      [--steps 3000] [--save-every 250] [--blocks-to-swap 24] [--dim 32] [--lr 1e-4] [--dit <file>] [--base-weights <file>|none]

Route B (lora_plan.md): --dit train/models/z_image_de_turbo_v1_bf16.safetensors --base-weights none.
Note (2026-10-03): the default --base-weights (ostris adapter v2) never loaded in v0, v1 or an earlier test run: its keys are
'diffusion_model.*.lora_A' and musubi's merge found 0 modules. Those LoRAs were plain Turbo training.

Outputs: train/output/<name>/<name>-stepNNNNNNNN.safetensors (musubi format), train/output/<name>/PROVENANCE.txt and
ComfyUI/models/loras/<name>_sNNNN.safetensors (converted, '--target other'). Log: logs/train_<name>.log
Last stdout line: 'DONE name=<name> exit=<training exit code> converted=<n>' (train/pilot/v2/after_train.py waits on it).
"""
import argparse, glob, hashlib, os, re, subprocess, sys, threading, time

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "ComfyUI_windows_portable"))
MUSUBI_ROOT = os.environ.get("MUSUBI_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "musubi-tuner"))
G = PROJECT_ROOT
MT = MUSUBI_ROOT
PY = rf"{MT}\.venv\Scripts\python.exe"
ACC = rf"{MT}\.venv\Scripts\accelerate.exe"
M = rf"{G}\train\models"
VAE = rf"{COMFY_ROOT}\ComfyUI\models\vae\z_image_ae.safetensors"
LORAS = rf"{COMFY_ROOT}\ComfyUI\models\loras"
sys.path.insert(0, rf"{G}\tools")
from run_edit import mem_sample  # noqa: E402


def run(cmd, log):
    env = dict(os.environ, PYTHONUTF8="1")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n"); fh.flush()
        return subprocess.run(cmd, cwd=MT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--blocks-to-swap", type=int, default=24)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--dit", default=rf"{M}\z_image_turbo_bf16.safetensors")
    ap.add_argument("--base-weights", default=rf"{M}\zimage_turbo_training_adapter_v2.safetensors", help="'none' = no adapter")
    ap.add_argument("--network-weights", default="", help="continue from these musubi-format LoRA weights (fresh optimizer, lr constant)")
    ap.add_argument("--step-offset", type=int, default=0, help="add to the step numbers of the converted files (continuation runs)")
    a = ap.parse_args()
    if a.network_weights:  # musubi runs with cwd = the musubi folder, so a relative path would not resolve (jzrth_v1c, 2026-10-06)
        a.network_weights = os.path.abspath(a.network_weights)
    out = rf"{G}\train\output\{a.name}"
    if glob.glob(rf"{out}\*.safetensors") or glob.glob(rf"{LORAS}\{a.name}_s*.safetensors"):
        print(f"DONE name={a.name} exit=refused converted=0 ({a.name} already has checkpoints; use a new --name)", flush=True); sys.exit(1)
    os.makedirs(out, exist_ok=True)
    log = rf"{G}\logs\train_{a.name}.log"
    ds = os.path.abspath(a.dataset)
    dh = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(os.path.dirname(ds), "*", "*.*"))):
        if not f.endswith(".safetensors"):
            dh.update(open(f, "rb").read())
    git = subprocess.run(["git", "-C", MT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(rf"{out}\PROVENANCE.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} dit={a.dit} ({os.path.getsize(a.dit)} B) base_weights={a.base_weights} dataset={ds} "
                 f"data_sha={dh.hexdigest()[:12]} musubi={git} argv={' '.join(sys.argv[1:])}\n")
    if not a.skip_cache:
        assert run([PY, r"src\musubi_tuner\zimage_cache_latents.py", "--dataset_config", ds, "--vae", VAE], log) == 0, "latent caching failed"
        assert run([PY, r"src\musubi_tuner\zimage_cache_text_encoder_outputs.py", "--dataset_config", ds, "--text_encoder",
                    rf"{M}\qwen_3_4b.safetensors", "--fp8_llm", "--batch_size", "1"], log) == 0, "text encoder caching failed"
    # memory log every 30 s while training (VRAM spill shows up as shared-memory growth on Windows)
    stop = threading.Event()
    memlog = rf"{G}\logs\train_{a.name}_memory.csv"

    def sampler():
        with open(memlog, "a", encoding="utf-8") as fh:
            while not stop.is_set():
                d, s, av, pf, cm = mem_sample()
                fh.write(f"{time.strftime('%H:%M:%S')},{d:.0f},{s:.2f},{av:.2f},{pf:.1f},{cm:.0f}\n"); fh.flush()
                stop.wait(30)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.time()
    bw = ([] if a.base_weights.lower() == "none" else ["--base_weights", a.base_weights]) + (["--network_weights", a.network_weights] if a.network_weights else [])
    code = run([ACC, "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16", r"src\musubi_tuner\zimage_train_network.py",
                "--dit", a.dit, "--vae", VAE, "--text_encoder", rf"{M}\qwen_3_4b.safetensors", *bw, "--dataset_config", ds,
                "--fp8_base", "--fp8_scaled", "--fp8_llm", "--blocks_to_swap", str(a.blocks_to_swap), "--gradient_checkpointing", "--split_attn", "--sdpa",
                "--timestep_sampling", "shift", "--weighting_scheme", "none", "--discrete_flow_shift", "2.0",
                "--optimizer_type", "adamw8bit", "--learning_rate", a.lr, "--network_module", "networks.lora_zimage",
                "--network_dim", str(a.dim), "--network_alpha", str(a.dim), "--max_train_steps", str(a.steps), "--save_every_n_steps", str(a.save_every),
                "--seed", str(42 + a.step_offset), "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
                "--output_dir", out, "--output_name", a.name, "--logging_dir", rf"{out}\logs"], log)
    stop.set()
    print(f"training exit {code} after {(time.time() - t0) / 3600:.2f} h", flush=True)
    n = 0
    for ck in sorted(glob.glob(rf"{out}\{a.name}*.safetensors")):
        m = re.search(r"step(\d+)", ck)
        step = (int(m.group(1)) if m else a.steps) + a.step_offset
        dest = rf"{LORAS}\{a.name}_s{step}.safetensors"
        if not os.path.exists(dest):
            run([PY, r"src\musubi_tuner\convert_lora.py", "--input", ck, "--output", dest, "--target", "other"], log)
            print(f"converted {os.path.basename(ck)} -> {dest}", flush=True)
        n += m is not None and os.path.exists(dest)
    print(f"DONE name={a.name} exit={code} converted={n}", flush=True)


if __name__ == "__main__":
    main()
