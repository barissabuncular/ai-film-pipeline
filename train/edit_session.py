"""One unattended edit session: wait for a marker in a log, start ComfyUI in edit mode, run several run_edit.py batches in order
(the edit model stays loaded between them), stop ComfyUI, then optionally start a night_char.py job (dry-run first).
A job 'i2i:<csv>' runs tools/run_img2img.py (Z-Image img2img) instead of run_edit.py; 'py:<script>@<arg>' runs any script that needs
ComfyUI up (e.g. the gated texture pass). --wait-pid waits for a process (e.g. the previous session) to exit before starting.
  python train/edit_session.py --wait logs/night_morvn_v1.log --marker "DONE columns|CHAIN_FAILED" \
      --jobs train/pilot/jazreth/w2.csv:train/pilot/jazreth/w2 train/pilot/pepin/w1.csv:train/pilot/pepin/w1 ... \
      --then -u train/night_char.py --name jzrth_v1 ...
Memory guard: tools/run_edit.py stops a batch by itself on low RAM / VRAM spill; the session then moves on to the next batch.
Log: logs/edit_session.log (one line per batch with edits and minutes). PIDs: logs/edit_session.pid, logs/comfy_edit.pid
"""
import argparse, os, re, subprocess, sys, time, urllib.request
import psutil

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_ROOT = os.environ.get("COMFY_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "ComfyUI_windows_portable"))
G = PROJECT_ROOT; C = COMFY_ROOT; PY = rf"{C}\python_embeded\python.exe"
LOG = rf"{G}\logs\edit_session.log"


def log(m):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + m + "\n")


def up():
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2); return True
    except Exception:
        return False


def kill(p):
    try:
        for c in p.children(recursive=True)[::-1] + [p]:
            try:
                c.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass


ap = argparse.ArgumentParser()
ap.add_argument("--wait", default=""); ap.add_argument("--marker", default="DONE"); ap.add_argument("--jobs", nargs="+", required=True)
ap.add_argument("--wait-pid", type=int, default=0)
ap.add_argument("--then", nargs=argparse.REMAINDER, default=[])
a = ap.parse_args()
if a.wait_pid and psutil.pid_exists(a.wait_pid):
    log(f"waiting for pid {a.wait_pid} to exit")
    while psutil.pid_exists(a.wait_pid):
        time.sleep(20)
    log(f"pid {a.wait_pid} exited")
open(rf"{G}\logs\edit_session.pid", "w").write(str(os.getpid()))
if a.wait:
    log(f"waiting for /{a.marker}/ in {a.wait}")
    while True:
        p = os.path.join(G, a.wait)
        if os.path.exists(p) and re.search(a.marker, open(p, encoding="utf-8", errors="replace").read()):
            break
        time.sleep(20)
    log("marker seen")
for _ in range(12):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    if not [x for x in out.splitlines() if re.search(r"python|accelerate", x, re.I)] and not up():
        break
    time.sleep(15)
else:
    log("GPU or port 8188 still busy after 3 min; nothing started"); sys.exit(1)
comfy = subprocess.Popen([PY, "-s", r"ComfyUI\main.py", "--windows-standalone-build", "--disable-auto-launch", "--reserve-vram", "1.5",
                          "--disable-pinned-memory", "--fast-disk"], cwd=C, stdout=open(rf"{G}\logs\comfy_edit_session.log", "a"), stderr=subprocess.STDOUT)
open(rf"{G}\logs\comfy_edit.pid", "w").write(str(comfy.pid))
for _ in range(120):
    if up():
        break
    time.sleep(2)
if not up():
    log("ComfyUI did not come up"); kill(psutil.Process(comfy.pid)); sys.exit(1)
log("ComfyUI up (edit mode)")
def job_cmd(j):
    """(arg file that must exist, command) for one job: 'py:<script>@<arg>', 'i2i:<csv>' or '<csv>:<out dir>' (run_edit.py)."""
    if j.startswith("py:"):  # e.g. py:train/texture_pass.py@train/pilot/cain/texture.json
        script, arg = j[3:].split("@", 1)
        return arg, [script, arg]
    if j.startswith("i2i:"):
        return j[4:], ["tools/run_img2img.py", "--csv", j[4:]]
    csvp, outd = j.split(":", 1)
    return csvp, ["tools/run_edit.py", "--csv", csvp, "--out-dir", outd]


try:  # ComfyUI is stopped even if a job or this loop crashes (2026-10-07: a crash left it running idle)
    for j in a.jobs:
        csvp, cmd = job_cmd(j)
        if not os.path.exists(os.path.join(G, csvp)):
            log(f"SKIP {csvp}: missing"); continue
        t0 = time.time(); name = os.path.basename(os.path.dirname(csvp)) + "_" + os.path.splitext(os.path.basename(csvp))[0]
        r = subprocess.run([PY, "-u", *cmd], cwd=G,
                           stdout=open(rf"{G}\logs\edit_{name}.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
        tail = open(rf"{G}\logs\edit_{name}.log", encoding="utf-8", errors="replace").read().strip().splitlines()[-1:]
        log(f"{csvp}: exit {r.returncode} in {(time.time() - t0) / 60:.1f} min; last line: {tail[0] if tail else ''}")
        if not up():
            log("ComfyUI stopped answering; ending the session"); break
except Exception as e:
    log(f"session failed: {type(e).__name__}: {e}")
    raise
finally:
    kill(psutil.Process(comfy.pid)); log("ComfyUI stopped")
if a.then:
    time.sleep(10)
    job = [x for x in a.then if x != "--"]
    r = subprocess.run([PY, *job, "--dry"], cwd=G, capture_output=True, text=True)
    if r.returncode:
        log(f"next job dry run failed ({r.returncode}); not started: {(r.stdout + r.stderr)[-400:]}"); sys.exit(1)
    p = subprocess.Popen([PY, *job], cwd=G, stdout=open(rf"{G}\logs\edit_session_next.out", "a"), stderr=subprocess.STDOUT,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    log(f"started next job pid {p.pid}: {' '.join(job)[:200]}")
log("DONE")
