"""Video quality test (2026-10-10). The user chose quality over GPU time for the video step; the video check (2026-10-09) showed Wan 2.2
i2v at 960x544 with the 4-step Lightning LoRAs softens every image by about 1 point. Same keyframes and seeds, several settings ("arms");
the middle and a late frame of each clip are upscaled like the post stage (4x NMKD-Siax -> 1920x1080) for a blind judge.
  python train/video_quality.py train/pilot/video_quality/arms.json        (starts and stops ComfyUI itself)
Arm keys: name, width, height, length, steps, split (high-noise steps), cfg, shift, lightning, reserve (--reserve-vram),
  only (shot ids, empty = all), if_failed (run only when that arm failed), copy_from (existing clips <dir>/<shot>/clip_<seed+100>.mp4).
An arm stops at its first error (e.g. out of memory). Writes <out>/<shot>/<arm>.mp4, <arm>_mid.png, <arm>_late.png and <out>/times.csv.
Log logs/video_quality.log, last line DONE.
"""
import csv, json, os, shutil, subprocess, sys, time, urllib.request
import cv2, psutil

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(G); sys.path.insert(0, G)
import run_shots as R  # noqa: E402
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(G), "ComfyUI_windows_portable"))
C = COMFY_ROOT; PY = sys.executable; HOST = "127.0.0.1:8188"
CFG = json.load(open(sys.argv[1], encoding="utf-8")); OUT = CFG["out"]; LOG = os.path.join(G, "logs", "video_quality.log")
DEADLINE = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + CFG["deadline"], "%Y-%m-%d %H:%M"))
DEADLINE += 86400 if DEADLINE < time.time() else 0  # HH:MM already passed today -> tomorrow


def log(m):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + m + "\n"); print(m, flush=True)


def up():
    try:
        urllib.request.urlopen(f"http://{HOST}/system_stats", timeout=2); return True
    except Exception:
        return False


def comfy_start(reserve):
    assert not up(), "ComfyUI already running"
    p = subprocess.Popen([PY, "-s", r"ComfyUI\main.py", "--windows-standalone-build", "--disable-auto-launch", "--reserve-vram", str(reserve)], cwd=C,
                         stdout=open(os.path.join(G, "logs", "comfy_video_quality.log"), "a"), stderr=subprocess.STDOUT)
    open(os.path.join(G, "logs", "comfy_video_quality.pid"), "w").write(str(p.pid))
    for _ in range(150):
        if up() or p.poll() is not None: break
        time.sleep(2)
    assert up(), f"ComfyUI did not come up (exit {p.poll()})"
    log(f"ComfyUI up (--reserve-vram {reserve})"); return p


def comfy_stop(p):
    if p is None: return
    try:
        pp = psutil.Process(p.pid)
        for c in pp.children(recursive=True)[::-1] + [pp]:
            try: c.kill()
            except psutil.Error: pass
    except psutil.Error:
        pass
    for _ in range(30):
        if not up(): break
        time.sleep(1)
    log("ComfyUI stopped")


def video_wf(arm, s, uploaded, seed):
    wf = R.load_wf("i2v_wan22_14b.json")
    pos, load, i2v = R.find_node(wf, "CLIPTextEncode", 0), R.find_node(wf, "LoadImage"), R.find_node(wf, "WanImageToVideo")
    k_hi, k_lo, save = R.find_node(wf, "KSamplerAdvanced", 0), R.find_node(wf, "KSamplerAdvanced", 1), R.find_node(wf, "SaveVideo")
    wf[pos]["inputs"]["text"] = s["motion_prompt"]; wf[load]["inputs"]["image"] = uploaded
    wf[i2v]["inputs"].update(width=arm["width"], height=arm["height"], length=arm["length"])
    for k in (k_hi, k_lo):
        wf[k]["inputs"].update(noise_seed=seed, steps=arm["steps"], cfg=arm["cfg"])
    wf[k_hi]["inputs"].update(start_at_step=0, end_at_step=arm["split"]); wf[k_lo]["inputs"].update(start_at_step=arm["split"], end_at_step=arm["steps"])
    for v in wf.values():
        if v["class_type"] == "ModelSamplingSD3": v["inputs"]["shift"] = arm["shift"]
    if not arm["lightning"]:  # full model: bypass both speed LoRAs
        for lid in [k for k, v in wf.items() if v["class_type"] == "LoraLoaderModelOnly"]:
            src = wf[lid]["inputs"]["model"]
            for v in wf.values():
                for n, val in v["inputs"].items():
                    if val == [lid, 0]: v["inputs"][n] = src
            del wf[lid]
    wf[save]["inputs"]["filename_prefix"] = f"GameStories/vq/{s['id']}_{arm['name']}"
    return wf


def wait_or_interrupt(pid, limit):
    """R.wait with a stall guard (2026-10-10: 1280x720x81 at --reserve-vram 3.0 spilled into shared memory and ran 20 min without one step)."""
    t0 = time.time()
    while True:
        e = R.api(HOST, f"/history/{pid}").get(pid)
        if e:
            return R.wait(HOST, pid, poll=1)
        if time.time() - t0 > limit:
            try:
                R.api(HOST, "/interrupt", data=b"{}", method="POST")
            except ValueError:
                pass  # empty reply body
            R.wait(HOST, pid, poll=5)  # raises: interrupted
            raise RuntimeError(f"stalled: no result after {limit / 60:.0f} min, interrupted")
        time.sleep(15)


def upscale(path_png, dest):
    up_name = R.upload_image(HOST, path_png, name=os.path.basename(path_png))
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": up_name}},
          "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}},
          "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
          "4": {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": 1920, "height": 1080, "crop": "center", "image": ["3", 0]}},
          "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "GameStories/vq/up", "images": ["4", 0]}}}
    outputs, _ = R.wait(HOST, R.submit(HOST, wf), poll=2)
    for o in outputs.values():
        for im in o.get("images", []):
            R.download(HOST, im, dest); return


def frames(mp4, d, name):
    cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); res = []
    for tag, i in (("mid", n // 2), ("late", int(n * 0.85))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read(); assert ok, f"cannot read frame {i} of {mp4}"
        raw = os.path.join(d, f"_{name}_{tag}_raw.png"); cv2.imwrite(raw, fr); res.append((raw, os.path.join(d, f"{name}_{tag}.png")))
    return res


shots = R.read_shots(CFG["shots"], ""); failed, comfy, cur_reserve = set(), None, None
times = open(os.path.join(OUT, "times.csv"), "a", newline="", encoding="utf-8"); tw = csv.writer(times)
log(f"start: {len(shots)} shots, arms {[a['name'] for a in CFG['arms']]}, deadline {CFG['deadline']}")
try:
    for arm in CFG["arms"]:
        if arm.get("if_failed") and arm["if_failed"] not in failed:
            continue
        out_name = arm.get("out_name", arm["name"])
        for s in shots:
            if arm.get("only") and s["id"] not in arm["only"]:
                continue
            d = os.path.join(OUT, s["id"]); os.makedirs(d, exist_ok=True); seed = int(s["seed"]) + 100
            mp4 = os.path.join(d, f"{out_name}.mp4")
            if arm.get("copy_from"):
                shutil.copyfile(os.path.join(arm["copy_from"], s["id"], f"clip_{seed}.mp4"), mp4)
            elif not os.path.exists(mp4):
                if time.time() + arm.get("est_min", 10) * 60 > DEADLINE:
                    log(f"{arm['name']} {s['id']}: skipped, not enough time before the deadline"); continue
                if cur_reserve != arm["reserve"] or comfy is None or comfy.poll() is not None:
                    comfy_stop(comfy); comfy = comfy_start(arm["reserve"]); cur_reserve = arm["reserve"]
                try:
                    uploaded = R.upload_image(HOST, os.path.join(d, "selected.png"), name=f"vq_{s['id']}.png")
                    outputs, dt = wait_or_interrupt(R.submit(HOST, video_wf(arm, s, uploaded, seed)), 3 * arm.get("est_min", 10) * 60)
                    vids = [v for o in outputs.values() for v in o.get("images", []) + o.get("video", []) + o.get("gifs", [])]
                    R.download(HOST, vids[0], mp4)
                    tw.writerow([s["id"], arm["name"], out_name, f"{dt / 60:.2f}"]); times.flush()
                    log(f"{arm['name']} {s['id']}: {dt / 60:.1f} min -> {mp4}")
                except Exception as e:
                    log(f"{arm['name']} {s['id']}: FAILED {str(e)[:300]}"); failed.add(arm["name"])
                    comfy_stop(comfy); comfy = None; break
            if comfy is None or comfy.poll() is not None:
                comfy = comfy_start(arm["reserve"] if not arm.get("copy_from") else 1.5); cur_reserve = arm["reserve"] if not arm.get("copy_from") else 1.5
            for raw, dest in frames(mp4, d, out_name):
                if not os.path.exists(dest):
                    upscale(raw, dest)
            log(f"{arm['name']} {s['id']}: frames upscaled")
    log("DONE")
except Exception as e:
    log(f"FAILED {e!r}")
    raise
finally:
    comfy_stop(comfy)
