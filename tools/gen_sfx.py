"""Generate sound effects and ambience locally with AudioLDM2 (cvssp/audioldm2-large via diffusers).

Reads sound/sfx_prompts.csv (file,seconds,prompt) and writes sound/<file>.wav (16 kHz).
Run with the audio venv (AUDIO_ROOT\\.venv):  python tools\\gen_sfx.py [--only sfx_helicopter] [--candidates 3]
Candidates are written as <file>_c1.wav ... ; the script keeps the one with the best CLAP score as <file>.wav.
"""
import argparse, csv, os
import numpy as np, scipy.io.wavfile as wavfile, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(HERE, "sound")
NEG = "music, melody, singing, voice, speech, low quality, distorted, noise burst"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(SOUND, "sfx_prompts.csv"))
    ap.add_argument("--model", default="cvssp/audioldm2-large")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    from diffusers import AudioLDM2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AudioLDM2Pipeline.from_pretrained(a.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    print("device:", device)

    only = set(a.only.split(",")) if a.only else None
    with open(a.prompts, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file") and (not only or r["file"] in only)]
    for r in rows:
        secs = float(r["seconds"])
        # AudioLDM2 tops out around 30 s per generation; longer beds get generated at 30 s and looped in the mix
        secs = min(secs, 30.0)
        gen = torch.Generator(device).manual_seed(a.seed)
        out = pipe(r["prompt"], negative_prompt=NEG, num_inference_steps=a.steps, audio_length_in_s=secs,
                   num_waveforms_per_prompt=a.candidates, generator=gen)
        audios = out.audios  # already ranked by CLAP score, best first
        for i, au in enumerate(audios, start=1):
            wavfile.write(os.path.join(SOUND, f"{r['file']}_c{i}.wav"), 16000, (np.clip(au, -1, 1) * 32767).astype(np.int16))
        wavfile.write(os.path.join(SOUND, f"{r['file']}.wav"), 16000, (np.clip(audios[0], -1, 1) * 32767).astype(np.int16))
        print(f"{r['file']}: {secs:.0f}s x{len(audios)} -> sound/{r['file']}.wav (best of {len(audios)})")


if __name__ == "__main__":
    main()
