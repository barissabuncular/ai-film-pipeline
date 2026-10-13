"""Generate narration lines locally with Qwen3-TTS VoiceDesign (voice built from a text description).

Reads sound/vo_lines.csv (file,text) and writes sound/<file>.wav (24 kHz).
Run with the qwen venv (AUDIO_ROOT\\.venv-qwen):
  python tools\\gen_voice_qwen.py [--instruct "..."] [--only vo_01,vo_02] [--outdir sound/tts_test/qwen]
  --instruct   free-form voice description (English works). Include pacing words like "slow, deliberate".
  --takes N    N takes per line (files <file>_t1.wav ...), pick manually. Default 1 -> <file>.wav
  --ref        optional reference wav: switches to the Base model and clones that voice instead of designing one
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(HERE, "sound")
DEFAULT_INSTRUCT = ("Deep, gravelly, tired male voice, late 40s, low pitch, slow and deliberate film narrator, "
                    "close to the microphone, low energy, quiet intensity, weary, long pauses between sentences.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default=os.path.join(SOUND, "vo_lines.csv"))
    ap.add_argument("--outdir", default=SOUND)
    ap.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    ap.add_argument("--language", default="English")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--ref", default=None, help="reference wav to clone (uses Qwen3-TTS Base model instead)")
    ap.add_argument("--ref-text", default=None, help="transcript of --ref (optional, improves cloning)")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attn", default="sdpa", help="sdpa | eager | flash_attention_2")
    a = ap.parse_args()

    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel

    if a.ref and "VoiceDesign" in a.model:
        a.model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    print("loading", a.model)
    model = Qwen3TTSModel.from_pretrained(a.model, device_map="cuda:0", dtype=torch.bfloat16,
                                          attn_implementation=a.attn)
    os.makedirs(a.outdir, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    with open(a.lines, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file") and (not only or r["file"] in only)]

    for r in rows:
        for t in range(1, a.takes + 1):
            torch.manual_seed(a.seed + t)
            t0 = time.time()
            if a.ref:
                kw = dict(text=r["text"], language=a.language, ref_audio=a.ref)
                if a.ref_text:
                    kw["ref_text"] = a.ref_text
                wavs, sr = model.generate_voice_clone(**kw)
            else:
                wavs, sr = model.generate_voice_design(text=r["text"], language=a.language, instruct=a.instruct)
            wav = wavs[0]
            name = f"{r['file']}_t{t}.wav" if a.takes > 1 else f"{r['file']}.wav"
            out = os.path.join(a.outdir, name)
            sf.write(out, wav, sr)
            dt = time.time() - t0
            print(f"{r['file']} take {t}: {len(wav)/sr:.1f}s audio in {dt:.1f}s -> {out}", flush=True)
    if torch.cuda.is_available():
        print("peak VRAM MiB:", int(torch.cuda.max_memory_allocated() / 2**20))


if __name__ == "__main__":
    main()
