"""Run a command as soon as a log file shows a marker line (keeps the GPU busy between jobs without waiting for a person).
  python train/queue_after.py --wait logs/edit_jazreth_w1.log --marker DONE --kill logs/comfy_edit.pid -- <python args of the next job>
Kills the processes in the --kill pid files (tree), checks that no Python/CUDA job holds the GPU, runs the next job with --dry
first (it must exit 0), then starts it detached. Log: logs/queue_after.log
"""
import argparse, os, re, subprocess, sys, time
import psutil

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "ComfyUI_windows_portable"))
G = PROJECT_ROOT; PY = rf"{COMFY_ROOT}\python_embeded\python.exe"
LOG = rf"{G}\logs\queue_after.log"


def log(m):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + m + "\n")


ap = argparse.ArgumentParser()
ap.add_argument("--wait", required=True); ap.add_argument("--marker", default="DONE"); ap.add_argument("--kill", nargs="*", default=[])
ap.add_argument("--timeout-h", type=float, default=6); ap.add_argument("--anywhere", action="store_true", help="marker is a regex anywhere in the log")
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
job = [x for x in a.rest if x != "--"]
log(f"waiting for '{a.marker}' in {a.wait}, then: {' '.join(job)}")
t0 = time.time()
while True:
    t = open(os.path.join(G, a.wait), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(G, a.wait)) else ""
    if (re.search(a.marker, t) if a.anywhere else re.search(rf"^{re.escape(a.marker)}", t, re.M)) or (not a.anywhere and "STOPPED" in t):
        break
    if time.time() - t0 > a.timeout_h * 3600:
        log("timeout, nothing started"); sys.exit(1)
    time.sleep(20)
log("marker seen" + (" (STOPPED)" if "STOPPED" in t else ""))
for pf in a.kill:
    try:
        p = psutil.Process(int(open(os.path.join(G, pf)).read().strip()))
        for c in p.children(recursive=True)[::-1] + [p]:
            try:
                c.kill()
            except psutil.Error:
                pass
        log(f"killed {pf} ({p.pid})")
    except (OSError, ValueError, psutil.Error) as e:
        log(f"kill {pf}: {e}")
time.sleep(10)
out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], capture_output=True, text=True).stdout
if [x for x in out.splitlines() if re.search(r"python|accelerate", x, re.I)]:
    log("a Python/CUDA job still holds the GPU; nothing started: " + out.replace("\n", "; ")); sys.exit(1)
r = subprocess.run([PY, *job, "--dry"], cwd=G, capture_output=True, text=True)
if r.returncode:
    log(f"dry run failed ({r.returncode}): {(r.stdout + r.stderr)[-500:]}"); sys.exit(1)
p = subprocess.Popen([PY, *job], cwd=G, stdout=open(rf"{G}\logs\queue_after_job.out", "a"), stderr=subprocess.STDOUT,
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
log(f"started job pid {p.pid}")
