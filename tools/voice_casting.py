"""Voice casting with Qwen3-TTS VoiceDesign (2026-10-12): one model load, N takes per speaker on a casting line.
Reads a JSON {"speakers": [{"name", "instruct", "line"}, ...]} and writes <outdir>/<name>_t<k>.wav plus <outdir>/casting_sheet.csv
(name, take, seed, seconds of audio, seconds to generate). Skips takes that already exist (resume after a "need pc").
Run with the qwen venv (GPU; never while ComfyUI is up):
  python tools\\voice_casting.py story\\diablo\\episodes\\ep01_the_butcher\\voices_v1.json sound\\ep01\\casting_v1 --takes 4
"""
import argparse, csv, json, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("outdir")
    ap.add_argument("--takes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    a = ap.parse_args()
    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel

    spk = json.load(open(a.spec, encoding="utf-8"))["speakers"]
    if a.only:
        keep = set(a.only.split(",")); spk = [s for s in spk if s["name"] in keep]
    os.makedirs(a.outdir, exist_ok=True)
    sheet = os.path.join(a.outdir, "casting_sheet.csv"); new = not os.path.exists(sheet)
    fh = open(sheet, "a", newline="", encoding="utf-8"); w = csv.writer(fh)
    if new:
        w.writerow(["name", "take", "seed", "audio_s", "gen_s"])
    print("loading", a.model, flush=True)
    model = Qwen3TTSModel.from_pretrained(a.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    for s in spk:
        for t in range(1, a.takes + 1):
            out = os.path.join(a.outdir, f"{s['name']}_t{t}.wav")
            if os.path.exists(out):
                continue
            seed = a.seed + t
            torch.manual_seed(seed); t0 = time.time()
            wavs, sr = model.generate_voice_design(text=s["line"], language="English", instruct=s["instruct"])
            sf.write(out, wavs[0], sr); dt = time.time() - t0
            w.writerow([s["name"], t, seed, round(len(wavs[0]) / sr, 2), round(dt, 1)]); fh.flush()
            print(f"{s['name']} take {t}: {len(wavs[0]) / sr:.1f}s audio in {dt:.1f}s", flush=True)
    print("peak VRAM MiB:", int(torch.cuda.max_memory_allocated() / 2**20), "\nDONE", flush=True)


if __name__ == "__main__":
    main()
