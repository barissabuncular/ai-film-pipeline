"""SeedVR2 test (2026-10-10, the user approved the download): does SeedVR2 3B (native ComfyUI nodes, int8) as the video upscaler beat
the current post upscaler (4x NMKD-Siax -> 1920x1080)? Input: the current-setting clips train/pilot/video_quality/<shot>/l4_960.mp4
(960x544, 81 frames); their NMKD frames already exist (l4_960_mid.png / _late.png). Starts and stops ComfyUI itself.
  python train/seedvr2_test.py [test folder, default train/pilot/video_quality] [--name seedvr2] [--denoise 1.0]   (2026-10-11: tuning arms)
Writes <shot>/seedvr2.mp4, seedvr2_mid.png, seedvr2_late.png (same frame indices as the NMKD frames: n//2 and int(n*0.85)),
<out>/seedvr2_times.csv and a flicker number per clip (mean abs change between consecutive frames, both upscalers). Log logs/seedvr2_test.log.
"""
import csv, glob, os, subprocess, sys, time, urllib.request
import cv2, numpy as np, psutil

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(G); sys.path.insert(0, G)
import run_shots as R  # noqa: E402
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(G), "ComfyUI_windows_portable"))
C = COMFY_ROOT; PY = sys.executable; HOST = "127.0.0.1:8188"; V = "train/pilot/video_quality"; NAME, DEN = "seedvr2", 1.0  # 2026-10-11: any test folder and tuning arms
_a = sys.argv[1:]
if _a and not _a[0].startswith("--"): V = _a.pop(0)
while _a:
    k, v = _a.pop(0), _a.pop(0)
    if k == "--name": NAME = v
    elif k == "--denoise": DEN = float(v)
LOG = os.path.join(G, "logs", "seedvr2_test.log")


def log(m):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + m + "\n"); print(m, flush=True)


def up():
    try:
        urllib.request.urlopen(f"http://{HOST}/system_stats", timeout=2); return True
    except Exception:
        return False


def wf(uploaded, prefix, n):
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
                                                    "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": DEN}},
        "11": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["10", 0], "temporal_overlap": ["8", 1]}},
        # 2026-10-11: the template's decode tiles (512 px, 64 frames) spilled 7.3 GB into shared memory on 8 GB and ran 2.5 h; smaller tiles
        "12": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["11", 0], "vae": ["5", 0], "tile_size": 256, "overlap": 64, "temporal_size": 16, "temporal_overlap": 4}},
        "13": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["12", 0], "original_resized_images": ["3", 0], "color_correction_method": "lab"}},
        "14": {"class_type": "ImageScale", "inputs": {"image": ["13", 0], "upscale_method": "lanczos", "width": 1920, "height": 1080, "crop": "center"}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": ["2", 2]}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}},
        "17": {"class_type": "ImageFromBatch", "inputs": {"image": ["14", 0], "batch_index": n // 2, "length": 1}},
        "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": prefix + "_mid"}},
        "19": {"class_type": "ImageFromBatch", "inputs": {"image": ["14", 0], "batch_index": int(n * 0.85), "length": 1}},
        "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": prefix + "_late"}},
    }


def wait_or_interrupt(pid, limit):
    t0 = time.time()
    while True:
        if R.api(HOST, f"/history/{pid}").get(pid):
            return R.wait(HOST, pid, poll=1)[0], time.time() - t0  # time from submit
        if time.time() - t0 > limit:
            try:
                R.api(HOST, "/interrupt", data=b"{}", method="POST")
            except ValueError:
                pass
            raise RuntimeError(f"stalled: no result after {limit / 60:.0f} min, interrupted")
        time.sleep(10)


def flicker(mp4):
    cap = cv2.VideoCapture(mp4); prev, d = None, []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: d.append(float(np.abs(g - prev).mean()))
        prev = g
    return float(np.mean(d)) if d else None


assert not up(), "ComfyUI already running"
comfy = subprocess.Popen([PY, "-s", r"ComfyUI\main.py", "--windows-standalone-build", "--disable-auto-launch", "--reserve-vram", "1.5"], cwd=C,
                         stdout=open(os.path.join(G, "logs", "comfy_seedvr2.log"), "a"), stderr=subprocess.STDOUT)
open(os.path.join(G, "logs", "comfy_seedvr2.pid"), "w").write(str(comfy.pid))
tw = csv.writer(open(f"{V}/{NAME}_times.csv", "a", newline="", encoding="utf-8"))
try:
    for _ in range(150):
        if up(): break
        time.sleep(2)
    assert up(), "ComfyUI did not come up"
    log("ComfyUI up")
    for d in sorted(x for x in glob.glob(f"{V}/e*") if os.path.isdir(x)):
        shot, src = os.path.basename(d), os.path.join(d, "l4_960.mp4")
        if os.path.exists(os.path.join(d, f"{NAME}_late.png")):
            continue
        n = int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            up_name = R.upload_image(HOST, src, name=f"sv_{shot}.mp4")
            outputs, dt = wait_or_interrupt(R.submit(HOST, wf(up_name, f"GameStories/sv/{NAME}_{shot}", n)), 25 * 60)
        except Exception as e:
            log(f"{shot}: FAILED {str(e)[:300]}"); break
        for nid, o in outputs.items():
            for f in o.get("images", []) + o.get("video", []) + o.get("gifs", []):
                name = {"16": f"{NAME}.mp4", "18": f"{NAME}_mid.png", "20": f"{NAME}_late.png"}.get(nid)
                if name: R.download(HOST, f, os.path.join(d, name))
        fl_sv = flicker(os.path.join(d, f"{NAME}.mp4")) if os.path.exists(os.path.join(d, f"{NAME}.mp4")) else None
        fl_src = flicker(src)
        tw.writerow([shot, f"{dt / 60:.2f}", fl_src, fl_sv])
        log(f"{NAME} {shot}: {dt / 60:.1f} min; flicker source {fl_src:.2f}, {NAME} {fl_sv if fl_sv is None else round(fl_sv, 2)}")
    log("DONE")
finally:
    p = psutil.Process(comfy.pid)
    for c in p.children(recursive=True)[::-1] + [p]:
        try: c.kill()
        except psutil.Error: pass
    log("ComfyUI stopped")
