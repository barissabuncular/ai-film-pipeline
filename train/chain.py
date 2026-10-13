"""Run GPU jobs one after another, detached, each to its own log; a failed step is logged and the next one still runs.
  python train/chain.py train/pilot/chain_<date>.json        (JSON: {"name": ..., "wait_pid": optional, "steps": [{"log": "logs/x.log", "args": [...]}, ...]})
optional per-step "python": the interpreter for that step (default: this one; e.g. the qwen venv for TTS).
"wait_pid": start only after that process has ended (e.g. a job of an earlier chain that is still running).
An arg "{now+8h}" becomes the clock time (HH:MM) 8 hours after that step starts (relative deadlines; 2026-10-11).
Log logs/chain_<name>.log (last line DONE). Stop everything: taskkill /T /F /PID (Get-Content logs\\chain_<name>.pid)
"""
import datetime, json, os, re, subprocess, sys, time
import psutil

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(G)
C = json.load(open(sys.argv[1], encoding="utf-8")); LOG = os.path.join("logs", f"chain_{C['name']}.log")
open(os.path.join("logs", f"chain_{C['name']}.pid"), "w").write(str(os.getpid()))


def log(m):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + m + "\n")


if C.get("wait_pid") and psutil.pid_exists(C["wait_pid"]):
    log(f"waiting for PID {C['wait_pid']}"); psutil.Process(C["wait_pid"]).wait()
log(f"start {len(C['steps'])} steps")
for i, st in enumerate(C["steps"]):
    st["args"] = [re.sub(r"\{now\+([\d.]+)h\}", lambda m: (datetime.datetime.now() + datetime.timedelta(hours=float(m.group(1)))).strftime("%H:%M"), a)
                  for a in st["args"]]
    log(f"step {i + 1}: {' '.join(st['args'])}")
    with open(st["log"], "a", encoding="utf-8") as fh:
        r = subprocess.run([st.get("python", sys.executable), "-u", *st["args"]], stdout=fh, stderr=subprocess.STDOUT)
    log(f"step {i + 1} exit {r.returncode}")
log("DONE")
