"""Z-Image img2img over a list of images (the "harmonise" pass, plan step 6): re-renders each image at low denoise with
its own caption as the prompt, to clean edit-model skin texture without moving the face or the scene.

CSV columns: file, prompt, out (output path), denoise (optional, default --denoise), seed (optional).
  python tools/run_img2img.py --csv train/pilot/v1/harmonise.csv [--denoise 0.25]
"""
import argparse, csv, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from run_shots import submit, wait, download, upload_image, load_wf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--denoise", type=float, default=0.25); ap.add_argument("--host", default="127.0.0.1:8188")
a = ap.parse_args()
wf0 = load_wf("zimage_img2img.json")
for i, r in enumerate(csv.DictReader(open(os.path.join(HERE, a.csv), encoding="utf-8"))):
    wf = json.loads(json.dumps(wf0))
    wf["7"]["inputs"]["image"] = upload_image(a.host, os.path.join(HERE, r["file"]), name=f"h_{abs(hash(r['file'])) % 10**9}.png")
    wf["5"]["inputs"]["text"] = r["prompt"]
    wf["9"]["inputs"]["denoise"] = float(r.get("denoise") or a.denoise)
    wf["9"]["inputs"]["seed"] = int(r.get("seed") or 7000 + i)
    t0 = time.time()
    outputs, _ = wait(a.host, submit(a.host, wf), poll=1)
    dest = os.path.join(HERE, r["out"]); os.makedirs(os.path.dirname(dest), exist_ok=True)
    for node_out in outputs.values():
        for img in node_out.get("images", []):
            download(a.host, img, dest)
    print(f"[{i}] {r['file']} -> {r['out']} ({time.time() - t0:.0f}s)", flush=True)
print("DONE", flush=True)
