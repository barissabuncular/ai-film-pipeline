"""Generate narration lines locally with Chatterbox TTS.

Reads sound/vo_lines.csv (file,text) and writes sound/<file>.wav (24 kHz).
Run with the audio venv (AUDIO_ROOT\\.venv):  python tools\\gen_voice.py [--ref sound/ref_voice.wav] [--only vo_01,vo_02]
  --exaggeration 0.3   emotion intensity (0.25 to 0.5 for a flat, tired read; higher = more dramatic)
  --cfg 0.3            pacing/adherence; lower is slower and more deliberate
  --takes 2            generate N takes per line (files <file>_t1.wav, <file>_t2.wav), pick manually, copy to <file>.wav
"""
import argparse, csv, os, sys
import torch, soundfile as sf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(HERE, "sound")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default=os.path.join(SOUND, "vo_lines.csv"))
    ap.add_argument("--ref", default=None, help="10 to 20 s wav of the voice to clone; omit for the default voice")
    ap.add_argument("--exaggeration", type=float, default=0.3)
    ap.add_argument("--cfg", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = ChatterboxTTS.from_pretrained(device=device)

    only = set(a.only.split(",")) if a.only else None
    with open(a.lines, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file") and (not only or r["file"] in only)]
    for r in rows:
        for t in range(1, a.takes + 1):
            torch.manual_seed(a.seed + t)
            kw = dict(exaggeration=a.exaggeration, cfg_weight=a.cfg, temperature=a.temperature)
            if a.ref:
                kw["audio_prompt_path"] = a.ref
            wav = model.generate(r["text"], **kw)
            out = os.path.join(SOUND, f"{r['file']}_t{t}.wav")
            sf.write(out, wav.squeeze().detach().cpu().numpy(), model.sr)
            print(f"{r['file']} take {t}: {wav.shape[-1]/model.sr:.1f}s -> {os.path.relpath(out, HERE)}")
        if a.takes == 1:
            os.replace(os.path.join(SOUND, f"{r['file']}_t1.wav"), os.path.join(SOUND, f"{r['file']}.wav"))


if __name__ == "__main__":
    main()
