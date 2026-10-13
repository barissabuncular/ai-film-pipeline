"""Generate narration lines locally with IndexTTS-2.5 (zero-shot clone of a reference clip + emotion from text + speed control).

Reads sound/vo_lines.csv (file,text) and writes sound/<file>.wav (22.05 kHz).
Run with the index-tts uv venv (Python 3.11, created by `uv sync` in AUDIO_ROOT\\index-tts):
  python tools\\gen_voice_index.py --ref sound/ref_voice.wav [--emo-text "tired, weary, calm"] [--outdir sound/tts_test/index]
  --ref        REQUIRED: 5 to 15 s wav of the voice to clone (e.g. a Qwen VoiceDesign output you liked)
  --emo-text   emotion description; omit to use the reference clip's emotion. --emo-alpha 0..1 strength.
  --emo-ref    optional second wav whose emotion is used instead of text
  --speed      duration factor, >1.0 is slower (1.15 = slow, weary narration), 1.0 = natural
  --version    2.5 (default, checkpoints/) or 2 (checkpoints_2/)
"""
import argparse, csv, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(HERE, "sound")
AUDIO_ROOT = os.environ.get("AUDIO_ROOT", os.path.join(os.path.dirname(HERE), "audio"))
REPO = os.path.join(AUDIO_ROOT, "index-tts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default=os.path.join(SOUND, "vo_lines.csv"))
    ap.add_argument("--outdir", default=SOUND)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--emo-text", default=None)
    ap.add_argument("--emo-ref", default=None)
    ap.add_argument("--emo-alpha", type=float, default=0.6)
    ap.add_argument("--emo-vector", default=None,
                    help="8 floats: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm (0..1). "
                         "Skips the 1.2 GB Qwen emotion classifier that --emo-text needs; saves VRAM on 8 GB cards. "
                         "Weary-narrator preset: 0,0,0.03,0,0,0.6,0,0")
    ap.add_argument("--speed", type=float, default=1.15, help="duration factor; >1 slower (v2.5 only)")
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--version", default="2.5", choices=["2.5", "2"])
    ap.add_argument("--fp32", action="store_true", help="disable bf16/fp16")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    # resolve paths before chdir into the repo (IndexTTS expects to run from its own folder)
    a.outdir = os.path.abspath(a.outdir)
    a.lines = os.path.abspath(a.lines)
    a.ref = os.path.abspath(a.ref)
    if a.emo_ref:
        a.emo_ref = os.path.abspath(a.emo_ref)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    half = not a.fp32
    emo_vec = [float(x) for x in a.emo_vector.split(",")] if a.emo_vector else None
    if emo_vec:
        a.emo_text = None  # vector wins; do not load the Qwen emotion classifier
    if a.version == "2.5":
        from indextts.infer_v2_5 import IndexTTS2
        ck = os.path.join(REPO, "checkpoints")
        tts = IndexTTS2(cfg_path=os.path.join(ck, "config.yaml"), model_dir=ck, use_bf16=half,
                        use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=bool(a.emo_text))
    else:
        from indextts.infer_v2 import IndexTTS2
        ck = os.path.join(REPO, "checkpoints_2")
        tts = IndexTTS2(cfg_path=os.path.join(ck, "config.yaml"), model_dir=ck, use_fp16=half,
                        use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=bool(a.emo_text))

    os.makedirs(a.outdir, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    with open(a.lines, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file") and (not only or r["file"] in only)]
    ref = os.path.abspath(os.path.join(HERE, a.ref)) if not os.path.isabs(a.ref) else a.ref
    for r in rows:
        for t in range(1, a.takes + 1):
            torch.manual_seed(a.seed + t)
            t0 = time.time()
            name = f"{r['file']}_t{t}.wav" if a.takes > 1 else f"{r['file']}.wav"
            outp = os.path.abspath(os.path.join(a.outdir, name))
            kw = dict(spk_audio_prompt=ref, text=r["text"], output_path=outp, emo_alpha=a.emo_alpha,
                      use_random=False, verbose=False)
            if a.version == "2.5":
                kw["lang"] = a.lang
                kw["duration_factor"] = a.speed
            if a.emo_ref:
                kw["emo_audio_prompt"] = os.path.abspath(a.emo_ref)
            elif emo_vec:
                kw["emo_vector"] = emo_vec
            elif a.emo_text:
                kw["use_emo_text"] = True
                kw["emo_text"] = a.emo_text
            tts.infer(**kw)
            print(f"{r['file']} take {t}: {time.time()-t0:.1f}s -> {outp}", flush=True)
    print("peak VRAM MiB:", int(torch.cuda.max_memory_allocated() / 2**20))


if __name__ == "__main__":
    main()
