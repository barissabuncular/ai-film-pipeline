"""Batch runner for the GameStories pipeline.

Reads shots.csv, submits keyframe and image-to-video jobs to a running ComfyUI,
waits for each, and downloads results into shots/<id>/.

Usage (from the project root):
  python run_shots.py keyframes            # N candidate stills per shot
  python run_shots.py video                # animate shots/<id>/selected.png (or first kf_*.png)
  python run_shots.py all
Options:
  --shots shots.csv   --takes 3   --candidates 4   --host 127.0.0.1:8188
  --only s01,s02      run a subset of shot ids
  --out-dir concepts/diablo   write results there instead of shots/ (concept art, look tests)
Keyframe rows may set width,height (default 1280x720) for portrait or square concept frames.
"""
import argparse, csv, json, os, random, shutil, sys, time, urllib.request, urllib.parse, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
WF_DIR = os.path.join(HERE, "workflows")
SHOTS_DIR = os.path.join(HERE, "shots")


def api(host, path, data=None, method=None):
    url = f"http://{host}{path}"
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())


def submit(host, prompt):
    body = json.dumps({"prompt": prompt, "client_id": "gamestories"}).encode()
    res = api(host, "/prompt", body)
    if res.get("node_errors"):
        raise RuntimeError(json.dumps(res["node_errors"], indent=1))
    return res["prompt_id"]


def wait(host, prompt_id, poll=5):
    t0 = time.time()
    while True:
        hist = api(host, f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI error: {msgs[-1][1].get('exception_message') if msgs else 'unknown'}")
            return entry["outputs"], time.time() - t0
        time.sleep(poll)


def download(host, fileinfo, dest):
    q = urllib.parse.urlencode({"filename": fileinfo["filename"], "subfolder": fileinfo.get("subfolder", ""), "type": fileinfo.get("type", "output")})
    with urllib.request.urlopen(f"http://{host}/view?{q}") as r, open(dest + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(dest + ".part", dest)  # a killed download never leaves a truncated file under the final name


def upload_image(host, path, name=None):
    boundary = uuid.uuid4().hex
    name = name or os.path.basename(path)
    with open(path, "rb") as f:
        content = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n").encode() + content + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"http://{host}/upload/image", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())["name"]


def load_wf(name):
    with open(os.path.join(WF_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def find_node(wf, class_type, nth=0):
    hits = [k for k, v in wf.items() if v["class_type"] == class_type]
    return hits[nth]


def read_shots(path, only):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("id")]
    if only:
        keep = set(only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    return rows


def stage_keyframes(host, shots, candidates, kf_workflow="keyframe_zimage.json", poll=5, keep_going=False):
    failed = []
    for s in shots:
        try:
            keyframe_shot(host, s, candidates, kf_workflow, poll)
        except Exception as e:
            if not keep_going:
                raise
            failed.append(s["id"]); print(f"[{s['id']}] FAILED: {e}", flush=True)
    if failed:
        print(f"FAILED {len(failed)}: {','.join(failed)}", flush=True)


def keyframe_shot(host, s, candidates, kf_workflow, poll):
    wf0 = load_wf(kf_workflow)
    pos = find_node(wf0, "CLIPTextEncode", 0)
    ks = find_node(wf0, "KSampler")
    save = find_node(wf0, "SaveImage")
    latent = find_node(wf0, "EmptySD3LatentImage")
    if True:
        out = os.path.join(SHOTS_DIR, s["id"]); os.makedirs(out, exist_ok=True)
        base_seed = int(s.get("seed") or random.randint(1, 2**31))
        for i in range(candidates):
            wf = json.loads(json.dumps(wf0))
            seed = base_seed + i
            wf[pos]["inputs"]["text"] = s["image_prompt"]
            wf[ks]["inputs"]["seed"] = seed
            if s.get("width") and s.get("height"):
                wf[latent]["inputs"]["width"] = int(s["width"]); wf[latent]["inputs"]["height"] = int(s["height"])
            # optional per-row LoRA (keyframe_zimage_lora.json): lora=<file> or "none" to bypass, lora_strength=<float>
            loras = [k for k, v in wf.items() if v["class_type"] == "LoraLoaderModelOnly"]
            if loras and s.get("lora"):
                if s["lora"].lower() == "none":
                    src = wf[loras[0]]["inputs"]["model"]
                    for v in wf.values():
                        for name, val in v["inputs"].items():
                            if val == [loras[0], 0]:
                                v["inputs"][name] = src
                    del wf[loras[0]]
                else:
                    wf[loras[0]]["inputs"]["lora_name"] = s["lora"]
                    if s.get("lora_strength"):
                        wf[loras[0]]["inputs"]["strength_model"] = float(s["lora_strength"])
                    if s.get("lora_steps"):  # 2026-10-10: LoRA only for the first N steps, plain model for the rest (identity early, texture late)
                        k, ms = int(s["lora_steps"]), wf[ks]["inputs"]["model"][0]
                        wf["ms_base"] = json.loads(json.dumps(wf[ms])); wf["ms_base"]["inputs"]["model"] = wf[loras[0]]["inputs"]["model"]
                        ki = wf[ks]["inputs"]; common = {a: ki[a] for a in ("positive", "negative", "steps", "cfg", "sampler_name", "scheduler")}
                        wf[ks] = {"class_type": "KSamplerAdvanced", "inputs": dict(common, model=ki["model"], latent_image=ki["latent_image"], add_noise="enable",
                                  noise_seed=seed, start_at_step=0, end_at_step=k, return_with_leftover_noise="enable")}
                        wf["ks_base"] = {"class_type": "KSamplerAdvanced", "inputs": dict(common, model=["ms_base", 0], latent_image=[ks, 0], add_noise="disable",
                                         noise_seed=seed, start_at_step=k, end_at_step=10000, return_with_leftover_noise="disable")}
                        for v in wf.values():
                            if v["class_type"] == "VAEDecode" and v["inputs"]["samples"] == [ks, 0]:
                                v["inputs"]["samples"] = ["ks_base", 0]
            wf[save]["inputs"]["filename_prefix"] = f"GameStories/{s['id']}/kf"
            pid = submit(host, wf)
            outputs, dt = wait(host, pid, poll)
            for node_out in outputs.values():
                for img in node_out.get("images", []):
                    dest = os.path.join(out, f"kf_{seed}.png")
                    download(host, img, dest)
                    print(f"[{s['id']}] keyframe seed={seed} {dt:.0f}s -> {os.path.relpath(dest, HERE)}")


def pick_keyframe(shot_dir):
    sel = os.path.join(shot_dir, "selected.png")
    if os.path.exists(sel):
        return sel
    kfs = sorted(f for f in os.listdir(shot_dir) if f.startswith("kf_") and f.endswith(".png"))
    if not kfs:
        raise FileNotFoundError(f"no keyframes in {shot_dir}; run keyframes stage first")
    return os.path.join(shot_dir, kfs[0])


def stage_video(host, shots, takes, seed_offset=0):
    wf0 = load_wf("i2v_wan22_14b.json")
    pos = find_node(wf0, "CLIPTextEncode", 0)
    load = find_node(wf0, "LoadImage")
    k_hi = find_node(wf0, "KSamplerAdvanced", 0)
    k_lo = find_node(wf0, "KSamplerAdvanced", 1)
    i2v = find_node(wf0, "WanImageToVideo")
    save = find_node(wf0, "SaveVideo")
    for s in shots:
        out = os.path.join(SHOTS_DIR, s["id"])
        kf = pick_keyframe(out)
        uploaded = upload_image(host, kf, name=f"{s['id']}_selected.png")
        base_seed = int(s.get("seed") or random.randint(1, 2**31))
        for t in range(takes):
            wf = json.loads(json.dumps(wf0))
            seed = base_seed + 100 + seed_offset + t
            wf[pos]["inputs"]["text"] = s["motion_prompt"]
            wf[load]["inputs"]["image"] = uploaded
            if s.get("width"):
                wf[i2v]["inputs"]["width"] = int(s["width"])
            if s.get("height"):
                wf[i2v]["inputs"]["height"] = int(s["height"])
            if s.get("length"):  # 2026-10-12: frames per clip (4n+1, 16 fps); default 81 = 5 s
                wf[i2v]["inputs"]["length"] = int(s["length"])
            wf[k_hi]["inputs"]["noise_seed"] = seed
            wf[k_lo]["inputs"]["noise_seed"] = seed
            wf[save]["inputs"]["filename_prefix"] = f"GameStories/{s['id']}/clip"
            pid = submit(host, wf)
            outputs, dt = wait(host, pid, poll=15)
            for node_out in outputs.values():
                for vid in node_out.get("images", []) + node_out.get("video", []) + node_out.get("gifs", []):
                    dest = os.path.join(out, f"clip_{seed}.mp4")
                    download(host, vid, dest)
                    print(f"[{s['id']}] take {t+1}/{takes} seed={seed} {dt/60:.1f} min -> {os.path.relpath(dest, HERE)}")


def stage_post(host, shots):
    """Upscale + interpolate shots/<id>/selected.mp4 (or first clip_*.mp4) to shots/<id>/final_1080p.mp4."""
    wf0 = load_wf("post_upscale_rife.json")
    load = find_node(wf0, "LoadVideo")
    save = find_node(wf0, "SaveVideo")
    for s in shots:
        out = os.path.join(SHOTS_DIR, s["id"])
        sel = os.path.join(out, "selected.mp4")
        if not os.path.exists(sel):
            clips = sorted(f for f in os.listdir(out) if f.startswith("clip_") and f.endswith(".mp4"))
            if not clips:
                print(f"[{s['id']}] no clip, skipping"); continue
            sel = os.path.join(out, clips[0])
        uploaded = upload_image(host, sel, name=f"{s['id']}_selected.mp4")
        wf = json.loads(json.dumps(wf0))
        wf[load]["inputs"]["file"] = uploaded
        wf[save]["inputs"]["filename_prefix"] = f"GameStories/{s['id']}/final"
        pid = submit(host, wf)
        outputs, dt = wait(host, pid, poll=10)
        for node_out in outputs.values():
            for vid in node_out.get("images", []) + node_out.get("video", []) + node_out.get("gifs", []):
                dest = os.path.join(out, "final_1080p.mp4")
                download(host, vid, dest)
                print(f"[{s['id']}] post {dt/60:.1f} min -> {os.path.relpath(dest, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["keyframes", "video", "post", "all"])
    ap.add_argument("--shots", default=os.path.join(HERE, "shots.csv"))
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed-offset", type=int, default=0, help="shift video take seeds, to get fresh takes on a rerun")
    ap.add_argument("--kf-workflow", default="keyframe_zimage.json", help="keyframe workflow file in workflows/ (e.g. keyframe_zimage_lora.json)")
    ap.add_argument("--out-dir", default="", help="results folder instead of shots/ (relative to the project)")
    ap.add_argument("--poll", type=float, default=5, help="seconds between ComfyUI history polls (keyframes)")
    ap.add_argument("--keep-going", action="store_true", help="keyframes: log a failing shot and continue with the next")
    a = ap.parse_args()
    if a.out_dir:
        global SHOTS_DIR
        SHOTS_DIR = os.path.join(HERE, a.out_dir)
    shots = read_shots(a.shots, a.only)
    if not shots:
        sys.exit("no shots found")
    if a.stage in ("keyframes", "all"):
        stage_keyframes(a.host, shots, a.candidates, a.kf_workflow, a.poll, a.keep_going)
    if a.stage in ("video", "all"):
        stage_video(a.host, shots, a.takes, a.seed_offset)
    if a.stage == "post":
        stage_post(a.host, shots)


if __name__ == "__main__":
    main()
