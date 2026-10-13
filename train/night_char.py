"""One character-LoRA night, any character (generalised from train/pilot/v2/night_v2.py, the Aidan v2 night that passed).
Route B: De-Turbo DiT, no adapter. One detached process: train -> guard -> eval -> sheets -> blind judge copies.

  python train/night_char.py --name bchr_v1 --dataset train/datasets/bchr_v1/dataset.toml --steps 2500 --deadline 07:30
      --eval-csv train/pilot/butcher/eval_bchr_v1.csv --cols base,c1500,c2000,c2500,c1000 --label "base=no LoRA" ...
      [--train-stop HH:MM] [--review train/pilot/butcher/review] [--front e01,e03 --left e11] [--dry] [--eval-only]

- Guards (every 30 s, whole night): shared GPU memory 1.5 GB above each GPU process's settled level (ComfyUI's pinned
  memory counts as 'shared'), RAM under 1 GB, commit over 92% (each for 90 s), steps over twice the measured speed for
  10 min, the train-stop time and the deadline. A memory or contention stop skips the eval (--eval-only later).
- Eval rows render column by column in --cols order (put the main columns first), only for LoRA files that exist.
  A no-LoRA control of the first LoRA row checks that the LoRA really changes the picture.
- Outputs: train/pilot/eval_<name>/, sheets <review>/3_eval_<name>_seed*.jpg, blind copies train/pilot/eval_<name>_blind
  (key in logs/eval_<name>_blind_key.json). Log logs/night_<name>.log, last line DONE or CHAIN_FAILED.
  Stop everything: taskkill /T /F /PID (Get-Content logs\\night_<name>.pid)
"""
import argparse, csv, datetime, glob, hashlib, json, os, re, shutil, subprocess, sys, threading, time, traceback, urllib.request
import psutil
from PIL import Image, ImageChops

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "ComfyUI_windows_portable"))
MUSUBI_ROOT = os.environ.get("MUSUBI_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "musubi-tuner"))
G = PROJECT_ROOT; C = COMFY_ROOT; PY = rf"{C}\python_embeded\python.exe"
MT = MUSUBI_ROOT; MPY = rf"{MT}\.venv\Scripts\python.exe"; LORAS = rf"{C}\ComfyUI\models\loras"
DETURBO = rf"{G}\train\models\z_image_de_turbo_v1_bf16.safetensors"
SEC_PER_IMAGE = 7.0  # measured with --poll 1, Aidan v2 eval 2026-10-04
GUARD = {"reason": None}
GPU_PROCS = {}
A = None  # parsed args


def log(m):
    with open(rf"{G}\logs\night_{A.name}.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + m + "\n")


class Fail(Exception):
    pass


def at(hhmm):
    t = datetime.datetime.combine(datetime.date.today(), datetime.time(*map(int, hhmm.split(":"))))
    return (t if t > datetime.datetime.now() else t + datetime.timedelta(days=1)).timestamp()


def up():
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2); return True
    except Exception:
        return False


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def code(r):
    return r["id"].rsplit("_", 1)[1]


def png(base, r):
    return rf"{base}\{r['id']}\kf_{r['seed']}.png"


def same_pixels(a, b):
    return ImageChops.difference(Image.open(a).convert("RGB"), Image.open(b).convert("RGB")).getbbox() is None


def tail(path, n=4000):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - n)); return f.read().decode("utf-8", "replace")


def kill_tree(p, why):
    """Kill p and its descendants, leaves first; True when none is left (venv launchers use kill-on-close job objects)."""
    try:
        procs = p.children(recursive=True)
    except psutil.Error:
        procs = []
    for c in procs[::-1] + [p]:
        try:
            c.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as e:
            log(f"kill {c.pid} failed: {e}")
    _, alive = psutil.wait_procs(procs + [p], timeout=15)
    log(f"killed {p.pid} + {len(procs)} children ({why})" + (f"; STILL ALIVE {[x.pid for x in alive]}" if alive else ""))
    return not alive


def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    return not [x for x in out.splitlines() if re.search(r"python|accelerate", x, re.I)]


def convert_leftovers():
    for ck in sorted(glob.glob(rf"{G}\train\output\{A.name}\{A.name}-step*.safetensors")):
        dest = rf"{LORAS}\{A.name}_s{int(re.search(r'step(\d+)', ck).group(1)) + A.step_offset}.safetensors"
        if not os.path.exists(dest):
            subprocess.run([MPY, r"src\musubi_tuner\convert_lora.py", "--input", ck, "--output", dest, "--target", "other"], cwd=MT,
                           stdout=open(rf"{G}\logs\train_{A.name}.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
            log(f"converted {os.path.basename(ck)} -> {os.path.basename(dest)}" + ("" if os.path.exists(dest) else " FAILED"))


def guard(deadline, stop_evt):
    sys.path.insert(0, rf"{G}\tools")
    from run_edit import mem_sample
    hi = lo = hc = 0; seen, base, warm = None, None, []
    with open(rf"{G}\logs\night_{A.name}_memory.csv", "a", encoding="utf-8") as fh:
        while not stop_evt.is_set():
            d, s, av, pf, cm = mem_sample()
            fh.write(f"{time.strftime('%H:%M:%S')},{d:.0f},{s:.2f},{av:.2f},{pf:.1f},{cm:.0f}\n"); fh.flush()
            if set(GPU_PROCS) != seen:
                seen, base, warm = set(GPU_PROCS), None, []
            if base is None and s == s:
                warm.append(s); base = max(warm) if len(warm) >= 3 else None
            hi = hi + 1 if base is not None and s == s and s > base + 1.5 else 0
            lo = lo + 1 if av == av and av < 1.0 else 0
            hc = hc + 1 if cm == cm and cm > 92 else 0
            why = (f"VRAM spill: shared GPU memory {s:.2f} GB, {s - base:.2f} GB above its settled level, for 90 s" if hi >= 3
                   else f"available RAM {av:.1f} GB for 90 s" if lo >= 3 else f"commit charge {cm:.0f}% for 90 s" if hc >= 3
                   else "deadline reached" if time.time() > deadline else None)
            if why and not GUARD["reason"]:
                GUARD["reason"] = why; log("GUARD " + why + ": stopping GPU work")
            if GUARD["reason"]:
                for n, p in list(GPU_PROCS.items()):
                    if p.is_running():
                        kill_tree(p, GUARD["reason"])
            stop_evt.wait(30)


def train(train_stop):
    drvlog = rf"{G}\logs\train_{A.name}_driver.log"
    cmd = [PY, "-u", r"train\train_character.py", "--name", A.name, "--dataset", A.dataset, "--steps", str(A.steps),
           "--save-every", str(A.save_every), "--dit", DETURBO, "--base-weights", "none"] + (["--skip-cache"] if A.skip_cache else []) \
        + (["--network-weights", A.network_weights, "--step-offset", str(A.step_offset)] if A.network_weights else [])
    drv = subprocess.Popen(cmd, cwd=G, stdout=open(drvlog, "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    open(rf"{G}\logs\train_{A.name}_driver.pid", "w").write(str(drv.pid))
    log(f"training started, driver pid {drv.pid}: {' '.join(cmd[2:])}")
    trainer, slow, stopped, t0 = psutil.Process(drv.pid), 0, False, time.time()
    while drv.poll() is None:
        time.sleep(30)
        try:
            acc = [k for k in trainer.children(recursive=False) if k.name().lower().startswith("accelerate")]
        except psutil.Error:
            acc = []
        if acc:
            GPU_PROCS["trainer"] = acc[0]
        m = re.findall(r"(\d+\.\d+)s/it", tail(rf"{G}\logs\train_{A.name}.log"))
        slow = slow + 1 if m and float(m[-1]) > 2 * A.sec_per_step else 0
        if slow == 4:
            log(f"SLOW: {m[-1]} s/step for 2 min (expected {A.sec_per_step}); GPU or RAM contention?")
        why = (GUARD["reason"] or ("train-stop time reached" if time.time() > train_stop else None)
               or (f"steps at {m[-1]} s/step for 10 min (contention)" if slow >= 20 else None))
        if why and not stopped and "trainer" in GPU_PROCS:
            log(f"stopping training: {why}"); kill_tree(GPU_PROCS["trainer"], why); stopped = True
            if slow >= 20:
                GUARD["reason"] = why
    GPU_PROCS.pop("trainer", None)
    done = re.findall(rf"DONE name={A.name} exit=(\S+) converted=(\d+)", open(drvlog, encoding="utf-8", errors="replace").read())
    log(f"training finished after {(time.time() - t0) / 3600:.2f} h: " + (f"exit={done[-1][0]} converted={done[-1][1]}" if done else "no DONE line"))
    if done and done[-1][0] == "refused":
        raise Fail(f"train_character.py refused: {A.name} already has checkpoints")
    if not stopped and (not done or (done[-1][0] != "0" and done[-1][1] == "0")):
        raise Fail(f"training failed before its first checkpoint ({'exit ' + done[-1][0] if done else 'no DONE line'}); see logs/train_{A.name}.log")
    convert_leftovers()


def render(rs, out, deadline, label):
    for attempt in (1, 2):
        todo = sorted({r["id"] for r in rs if not os.path.exists(png(out, r))})
        if not todo or GUARD["reason"]:
            break
        tmp = rf"{G}\logs\eval_{A.name}_todo.csv"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(r for r in rs if r["id"] in todo)
        t0, n = time.time(), sum(1 for r in rs if r["id"] in todo)
        try:
            rc = subprocess.run([PY, "-u", "run_shots.py", "keyframes", "--kf-workflow", "keyframe_zimage_lora.json", "--shots", tmp,
                                 "--out-dir", os.path.relpath(out, G), "--candidates", "1", "--poll", "1", "--keep-going"], cwd=G,
                                stdout=open(rf"{G}\logs\eval_{A.name}.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT,
                                timeout=max(60, min(n * 40 + 120, deadline - time.time()))).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        k = sum(1 for r in rs if r["id"] in todo and os.path.exists(png(out, r)))
        log(f"{label} try {attempt}: {k}/{n} images in {(time.time() - t0) / 60:.1f} min ({(time.time() - t0) / max(1, k):.1f} s/image), exit {rc}")
        if k and (time.time() - t0) / k > 2.5 * SEC_PER_IMAGE:
            log(f"SLOW: {(time.time() - t0) / k:.1f} s/image (measured {SEC_PER_IMAGE}); GPU or RAM contention?")
        if GUARD["reason"]:
            break
        if not up():
            raise Fail("ComfyUI stopped answering")
    return [r["id"] + "/" + r["seed"] for r in rs if not os.path.exists(png(out, r))]


def comfy_start():
    for _ in range(8):
        if gpu_free():
            break
        time.sleep(15)
    else:
        raise Fail("another Python/CUDA job still holds the GPU; not starting ComfyUI")
    if up():
        raise Fail("port 8188 already answers: another ComfyUI is running")
    c = subprocess.Popen([PY, "-s", r"ComfyUI\main.py", "--windows-standalone-build", "--disable-auto-launch", "--reserve-vram", "1.5"],
                         cwd=C, stdout=open(rf"{G}\logs\comfy_eval_{A.name}.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    open(rf"{G}\logs\comfy_eval_{A.name}.pid", "w").write(str(c.pid)); GPU_PROCS["comfy"] = psutil.Process(c.pid)
    for _ in range(120):
        if up() or c.poll() is not None:
            break
        time.sleep(2)
    if not up():
        raise Fail(f"ComfyUI not up after 240 s (exit {c.poll()})")
    log("ComfyUI up")
    return c


def evaluate(deadline, out, cols):
    if GUARD["reason"]:
        raise Fail(f"guard: {GUARD['reason']}: no eval (run it later with --eval-only)")
    if time.time() > deadline - 10 * 60:
        raise Fail("less than 10 min to the deadline: no eval (run it later with --eval-only)")
    convert_leftovers()
    ev = [r for r in rows(A.eval_csv) if r["lora"] == "none" or os.path.exists(rf"{LORAS}\{r['lora']}")]
    have = [c for c in cols if any(code(r) == c for r in ev)]
    log(f"columns with a LoRA on disk: {have}")
    if not [r for r in ev if r["lora"] != "none"]:
        raise Fail("no checkpoint of this LoRA exists")
    comfy = comfy_start(); missing = []
    try:
        first = next(r for c in have for r in ev if code(r) == c and r["lora"] != "none")
        ctl = dict(first, id=first["id"].rsplit("_", 1)[0] + "_nolora", lora="none", lora_strength="")
        missing += render([first], out, deadline, "LoRA check") + render([ctl], out + "_ctl", deadline, "no-LoRA control")
        if all(os.path.exists(p) for p in (png(out, first), png(out + "_ctl", ctl))) and same_pixels(png(out, first), png(out + "_ctl", ctl)):
            raise Fail(f"{first['lora']} renders pixel-identical to no LoRA: it did not load")
        for c in have:
            if GUARD["reason"]:
                log(f"skipping {c}: {GUARD['reason']}"); continue
            missing += render([r for r in ev if code(r) == c], out, deadline, c)
    finally:
        p = GPU_PROCS.pop("comfy", None)
        ok = kill_tree(p, "eval over") if p and comfy.poll() is None else True
        log("ComfyUI stopped" if ok else "ComfyUI STILL RUNNING after kill")
    nk = len(re.findall(r"lora key not loaded", tail(rf"{G}\logs\comfy_eval_{A.name}.log", 10 ** 7)))
    if nk:
        log(f"WARNING: {nk} 'lora key not loaded' lines in comfy_eval_{A.name}.log")
    return missing, nk


def sheets(out, cols):
    cols = [c for c in cols if glob.glob(rf"{out}\*_{c}")]
    labels = sum([["--label", x] for x in A.label], [])
    r = subprocess.run([PY, "train/eval_sheet.py", "--dir", os.path.relpath(out, G), "--cols", ",".join(cols), "--thr", "0.45", "--name", A.name,
                        "--out", A.review, "--anchors", *A.anchors, *labels] + (["--front", A.front] if A.front else []) + (["--left", A.left] if A.left else []),
                       cwd=G, capture_output=True, text=True)
    log(r.stdout + r.stderr)
    if r.returncode:
        raise Fail(f"eval_sheet.py exit {r.returncode}")
    return cols


def blind(out, dest, cols, key_path):
    prompts = {(r["id"], r["seed"]): r["image_prompt"] for r in rows(A.eval_csv)}
    salt = os.urandom(8).hex()
    key = {f"X{i + 1:02d}": c for i, c in enumerate(sorted(cols, key=lambda c: hashlib.sha256(f"{salt}-{c}".encode()).hexdigest()))}
    shutil.rmtree(dest, ignore_errors=True); info = {}
    os.makedirs(rf"{dest}\_ref")
    for f in A.anchors:
        shutil.copyfile(os.path.join(G, f), rf"{dest}\_ref\{os.path.basename(f)}")
    for x, c in key.items():
        for d in glob.glob(rf"{out}\*_{c}"):
            scn = os.path.basename(d).rsplit("_", 1)[0]
            os.makedirs(rf"{dest}\{scn}_{x}")
            for f in glob.glob(rf"{d}\kf_*.png"):
                im = Image.open(f); im.load(); im.save(rf"{dest}\{scn}_{x}\{os.path.basename(f)}")  # drops ComfyUI metadata (LoRA name)
                info[f"{scn}_{x}/{os.path.basename(f)}"] = prompts.get((f"{scn}_{c}", os.path.basename(f)[3:-4]), "")
    json.dump(key, open(key_path, "w"), indent=1)
    for scn in sorted({k.rsplit("_X", 1)[0] for k in info}):
        json.dump({k: v for k, v in info.items() if k.startswith(scn + "_X")}, open(rf"{dest}\prompts_{scn}.json", "w", encoding="utf-8"), indent=1)
    log(f"blind folders: {len(info)} images under {list(key)}; key in {key_path}")


def dry(cols):
    problems = []
    need = [DETURBO, A.dataset, A.eval_csv] + [os.path.join(G, f) for f in A.anchors]
    problems += [f"missing {p}" for p in need if not os.path.exists(os.path.join(G, p))]
    ck = glob.glob(rf"{G}\train\output\{A.name}\*.safetensors") + glob.glob(rf"{LORAS}\{A.name}_s*.safetensors")
    if ck and not A.eval_only:
        problems.append(f"{A.name} checkpoints already exist")
    want = {f"{A.name}_s{s + A.step_offset}.safetensors" for s in range(A.save_every, A.steps + 1, A.save_every)}
    if A.network_weights and not os.path.exists(os.path.join(G, A.network_weights)):
        problems.append(f"missing --network-weights {A.network_weights}")
    ev = rows(A.eval_csv)
    problems += sorted({f"eval LoRA {r['lora']} will not exist" for r in ev if r["lora"] != "none" and r["lora"] not in want
                        and not os.path.exists(rf"{LORAS}\{r['lora']}")})
    problems += [f"--cols has {c}, which no eval row uses" for c in cols if not any(code(r) == c for r in ev)]
    problems += [f"eval column {c} is missing from --cols" for c in sorted({code(r) for r in ev} - set(cols))]
    problems += [f"no --label for {c}" for c in cols if not any(x.startswith(c + "=") for x in A.label)]
    if not gpu_free():
        problems.append("a Python/CUDA job is on the GPU")
    if up():
        problems.append("port 8188 answers: ComfyUI is running")
    r = subprocess.run([PY, "run_shots.py", "keyframes", "--shots", A.eval_csv, "--only", "__none__", "--poll", "1", "--keep-going"], cwd=G, capture_output=True, text=True)
    if "no shots found" not in r.stderr + r.stdout:
        problems.append("run_shots.py does not accept --poll/--keep-going")
    for p in problems:
        log("DRY PROBLEM " + p)
    log("dry run ok" if not problems else f"dry run: {len(problems)} problems")
    return not problems


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, required=True); ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--sec-per-step", type=float, default=6.6, help="expected training speed (Aidan v2: 6.56 at 640/1024)")
    ap.add_argument("--skip-cache", action="store_true", help="the dataset caches are already built")
    ap.add_argument("--network-weights", default="", help="continue from these musubi-format LoRA weights (new --name, --step-offset)")
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--deadline", required=True); ap.add_argument("--train-stop", default="")
    ap.add_argument("--eval-csv", required=True); ap.add_argument("--cols", required=True, help="eval columns, main ones first")
    ap.add_argument("--label", action="append", default=[], help="CODE=header text, one per column")
    ap.add_argument("--anchors", nargs="+", required=True, help="reference images for the sheets' face score and the judges")
    ap.add_argument("--review", required=True); ap.add_argument("--front", default=""); ap.add_argument("--left", default="")
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--eval-only", action="store_true")
    A = ap.parse_args()
    cols = A.cols.split(",")
    deadline = at(A.deadline)
    n_eval = len(rows(A.eval_csv)) + 2
    train_stop = at(A.train_stop) if A.train_stop else deadline - n_eval * SEC_PER_IMAGE - 15 * 60
    log(f"start {A.name}: {A.steps} steps, deadline {A.deadline}, training stops at the latest {time.strftime('%H:%M', time.localtime(train_stop))}; "
        f"expected training {A.steps * A.sec_per_step / 3600:.1f} h + eval {n_eval * SEC_PER_IMAGE / 60:.0f} min" + (" (DRY)" if A.dry else ""))
    if A.dry:
        sys.exit(0 if dry(cols) else 1)
    open(rf"{G}\logs\night_{A.name}.pid", "w").write(str(os.getpid()))
    stop = threading.Event(); threading.Thread(target=guard, args=(deadline, stop), daemon=True).start()
    out = rf"{G}\train\pilot\eval_{A.name}"
    try:
        if not A.eval_only:
            train(train_stop)
            if GUARD["reason"]:
                raise Fail(f"training was stopped ({GUARD['reason']}): no eval tonight; later: --eval-only")
        missing, nk = evaluate(deadline, out, cols)
        done_cols = sheets(out, cols)
        blind(out, rf"{G}\train\pilot\eval_{A.name}_blind", done_cols, rf"{G}\logs\eval_{A.name}_blind_key.json")
        log(f"DONE columns={','.join(done_cols)} missing={len(missing)}" + (f" ({missing[:5]})" if missing else "") +
            f" lora_keys_not_loaded={nk}" + (f" guard={GUARD['reason']}" if GUARD["reason"] else ""))
    except Fail as e:
        log(f"CHAIN_FAILED {e}")
    except Exception:
        log("CHAIN_FAILED " + traceback.format_exc())
    finally:
        stop.set()
        for n, p in list(GPU_PROCS.items()):
            if p.is_running():
                kill_tree(p, "chain end")


if __name__ == "__main__":
    main()
