"""Start ComfyUI, run one run_shots.py command, stop ComfyUI (2026-10-11; for chain steps such as the episode keyframes).
  python train/comfy_run.py keyframes --shots story/diablo/episodes/ep01_the_butcher/shots_ep01_v1.csv --kf-workflow keyframe_zimage_lora.json
      --candidates 4 --out-dir shots/ep01_v1 --keep-going
Everything after the script name goes to run_shots.py unchanged. If the first argument is a .py script (e.g. tools/run_edit.py),
that script runs instead with the remaining arguments (2026-10-12). Log: logs/comfy_run.log (ComfyUI), stdout = run_shots output.
"""
import os, subprocess, sys, time, urllib.request
import psutil
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(G), "ComfyUI_windows_portable"))
C = COMFY_ROOT; PY = sys.executable


def up():
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2); return True
    except Exception:
        return False


assert not up(), "ComfyUI already running"
comfy = subprocess.Popen([PY, "-s", r"ComfyUI\main.py", "--windows-standalone-build", "--disable-auto-launch", "--reserve-vram", "1.5"], cwd=C,
                         stdout=open(os.path.join(G, "logs", "comfy_run.log"), "a"), stderr=subprocess.STDOUT)
open(os.path.join(G, "logs", "comfy_run.pid"), "w").write(str(comfy.pid))
try:
    for _ in range(150):
        if up(): break
        time.sleep(2)
    assert up(), "ComfyUI did not come up"
    cmd = sys.argv[1:] if sys.argv[1].endswith(".py") else ["run_shots.py", *sys.argv[1:]]
    r = subprocess.run([PY, "-u", *cmd], cwd=G)
    print("run_shots exit", r.returncode, flush=True)
finally:
    p = psutil.Process(comfy.pid)
    for c in p.children(recursive=True)[::-1] + [p]:
        try: c.kill()
        except psutil.Error: pass
print("DONE", flush=True)
