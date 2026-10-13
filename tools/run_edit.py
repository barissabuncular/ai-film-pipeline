"""Batch image edits with Qwen-Image-Edit-2511 (workflows/qwen_edit_2511.json) on a running ComfyUI.

CSV columns: id, input, input2 (optional second reference), prompt, angles (strength of the multiple-angles LoRA,
0 = off), seed, count. Results: <out-dir>/<id>/ed_<seed>.png, plus <out-dir>/timings.csv (exact seconds per image
from ComfyUI's execution timestamps) and <out-dir>/memory.csv (every 5 s: dedicated VRAM, Windows shared GPU
memory, free RAM, pagefile).

Stop rules (the plan's window-1 guards): free RAM under 2 GB, or shared GPU memory rising more than 0.3 GB above
its level at start (on Windows a VRAM overflow spills into shared memory and shows up as slowness, not as an error).

  python tools/run_edit.py --csv train/pilot/w1_edits.csv --out-dir train/pilot/w1
"""
import argparse, csv, json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from run_shots import api, submit, download, upload_image, load_wf  # noqa: E402

STOP = {"reason": None}


def mem_sample():
    """(dedicated_mb, shared_gb, available_ram_gb, pagefile_pct). Available = free + standby cache (what Windows can hand out)."""
    try:
        ded = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0])
    except Exception:
        ded = float("nan")
    ps = ("$s=(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | Measure-Object -Property CookedValue -Sum;"
          "$av=(Get-Counter '\\Memory\\Available MBytes').CounterSamples[0].CookedValue;"
          "$pf=(Get-Counter '\\Paging File(_Total)\\% Usage').CounterSamples[0].CookedValue;"
          "$cm=(Get-Counter '\\Memory\\% Committed Bytes In Use').CounterSamples[0].CookedValue;"
          "Write-Output ('{0} {1} {2} {3}' -f $s.Sum, $av, $pf, $cm)")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20).stdout.split()
        shared = float(out[0]) / 1e9; free = float(out[1]) / 1024; pf = float(out[2]); cm = float(out[3])
    except Exception:
        shared = free = pf = cm = float("nan")
    return ded, shared, free, pf, cm


def monitor(path, stop_evt):
    base = None; pf0 = None; low_ram = high_shared = 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["time", "dedicated_mb", "shared_gb", "available_ram_gb", "pagefile_pct", "commit_pct"])
        while not stop_evt.is_set():
            ded, shared, free, pf, cm = mem_sample()
            if base is None and shared == shared:
                base = shared
            if pf0 is None and pf == pf:
                pf0 = pf
            w.writerow([time.strftime("%H:%M:%S"), round(ded), round(shared, 2), round(free, 2), round(pf, 1), round(cm)]); fh.flush()
            # Model loads and LoRA re-patching cause short dips (measured 2026-10-01: 0.3-0.8 GB available for
            # under 10 s, then 4.8 GB steady). Only sustained low memory, pagefile growth or a near-full commit stop the run.
            # A cold load of the edit stack sits at 0.6-0.7 GB available for 20-30 s with no paging (2026-10-01:
            # pagefile flat at 3%, commit 68%, ~5.5 GB of it clean memory-mapped model pages). Real trouble is
            # low memory together with paging, or a very long deep dip.
            paging = pf0 is not None and pf == pf and pf - pf0 > 3
            low_ram = low_ram + 1 if (free == free and (free < 0.4 or (free < 1.0 and paging))) else 0
            high_shared = high_shared + 1 if (base is not None and shared == shared and shared - base > 0.3) else 0
            if low_ram >= 6:
                STOP["reason"] = f"available RAM {free:.1f} GB with paging, for 30 s"
            if cm == cm and cm > 90:
                STOP["reason"] = f"commit charge {cm:.0f}% > 90%"
            # 2026-10-12: model swaps between edits spike shared memory to 6-9 GB for about 30 s (seen in memory.csv), so only a
            # spill that lasts about a minute stops the run (was 3 samples, which stopped the redo3 session at a model load)
            if high_shared >= 6:
                STOP["reason"] = f"shared GPU memory {shared - base:.2f} GB above start for about 60 s (VRAM spill)"
            if pf0 is not None and pf == pf and pf - pf0 > 10:
                STOP["reason"] = f"pagefile use rose {pf - pf0:.0f} points"
            stop_evt.wait(5)


def exec_seconds(host, pid):
    msgs = api(host, f"/history/{pid}")[pid]["status"].get("messages", [])
    ts = {m[0]: m[1].get("timestamp") for m in msgs}
    if ts.get("execution_start") and ts.get("execution_success"):
        return (ts["execution_success"] - ts["execution_start"]) / 1000
    return float("nan")


def wait_done(host, pid, poll=1):
    while True:
        hist = api(host, f"/history/{pid}")
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(msgs[-1][1].get("exception_message") if msgs else "ComfyUI error")
            return hist[pid]["outputs"]
        if STOP["reason"]:
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(f"http://{host}/interrupt", data=b"{}", method="POST")).read()
            raise RuntimeError(f"stopped: {STOP['reason']}")
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = os.path.join(HERE, a.out_dir); os.makedirs(out, exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, a.csv), encoding="utf-8")) if r.get("id")]
    if a.only:
        keep = set(a.only.split(",")); rows = [r for r in rows if r["id"] in keep]
    wf0 = load_wf("qwen_edit_2511.json")
    stop_evt = threading.Event()
    th = threading.Thread(target=monitor, args=(os.path.join(out, "memory.csv"), stop_evt), daemon=True); th.start()
    tf = open(os.path.join(out, "timings.csv"), "a", newline="", encoding="utf-8"); tw = csv.writer(tf)
    if tf.tell() == 0:
        tw.writerow(["id", "seed", "refs", "angles", "exec_s", "wall_s"])
    uploaded = {}
    try:
        for r in rows:
            for path in [r["input"]] + ([r["input2"]] if r.get("input2") else []):
                if path not in uploaded:
                    uploaded[path] = upload_image(a.host, os.path.join(HERE, path), name=f"gs_{abs(hash(path)) % 10**8}.png")
            d = os.path.join(out, r["id"]); os.makedirs(d, exist_ok=True)
            for i in range(int(r.get("count") or 1)):
                seed = int(r["seed"]) + i
                if os.path.exists(os.path.join(d, f"ed_{seed}.png")):
                    continue  # resume: this edit is already done
                wf = json.loads(json.dumps(wf0))
                wf["8"]["inputs"]["image"] = uploaded[r["input"]]
                if r.get("input2"):
                    wf["18"] = {"class_type": "LoadImage", "inputs": {"image": uploaded[r["input2"]]}}
                    wf["19"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["18", 0]}}
                    wf["10"]["inputs"]["image2"] = ["19", 0]; wf["11"]["inputs"]["image2"] = ["19", 0]
                ang = float(r.get("angles") or 0)
                if ang > 0:
                    wf["2"]["inputs"]["strength_model"] = ang
                else:
                    wf["3"]["inputs"]["model"] = ["1", 0]; del wf["2"]
                wf["10"]["inputs"]["prompt"] = r["prompt"]
                wf["15"]["inputs"]["seed"] = seed
                wf["17"]["inputs"]["filename_prefix"] = f"GameStories/edit/{r['id']}"
                t0 = time.time()
                pid = submit(a.host, wf)
                outputs = wait_done(a.host, pid)
                wall = time.time() - t0
                ex = exec_seconds(a.host, pid)
                for node_out in outputs.values():
                    for img in node_out.get("images", []):
                        dest = os.path.join(d, f"ed_{seed}.png"); download(a.host, img, dest)
                tw.writerow([r["id"], seed, 2 if r.get("input2") else 1, ang, round(ex, 1), round(wall, 1)]); tf.flush()
                print(f"[{r['id']}] seed={seed} exec {ex:.1f}s wall {wall:.1f}s -> {os.path.relpath(dest, HERE)}", flush=True)
    finally:
        stop_evt.set(); tf.close()
    print("DONE" if not STOP["reason"] else f"STOPPED: {STOP['reason']}", flush=True)


if __name__ == "__main__":
    main()
