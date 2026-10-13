"""Generate the trailer music bed locally with ACE-Step (3.5B, cpu offload for 8 GB cards).

Run with the ACE-Step venv (AUDIO_ROOT\\.venv-ace):  python tools\\gen_music.py [--seconds 60] [--takes 2] [--seed 42]
Writes sound/music_t1.wav, music_t2.wav ... (48 kHz). Copy the chosen take to sound/music.wav.
The first run downloads the 8.3 GB checkpoint to ~/.cache/ace-step.
"""
import argparse, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(HERE, "sound")

PROMPT = ("slow dark cinematic film score, sparse lonely solo piano motif in a minor key, cold sustained string pad, "
          "low cello, deep sub bass pulse, very slow tempo 60 bpm, restrained, melancholic, "
          "gradual tension build, a single hard orchestral hit near the end then a held piano note decaying to silence, "
          "instrumental, no drums, no vocals, film soundtrack, high quality")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()

    # torchaudio 2.11 delegates save() to torchcodec, which isn't installed; route saves through soundfile instead
    import torchaudio, soundfile as sf, torch

    def _save(path, wav, sr=None, *args, sample_rate=None, **kwargs):
        rate = int(sr if sr is not None else sample_rate)
        w = wav.detach().cpu().float()
        sf.write(str(path), w.T.numpy() if w.dim() == 2 else w.numpy(), rate)
    torchaudio.save = _save
    try:
        import torchaudio._torchcodec as _tc
        _tc.save_with_torchcodec = _save
    except Exception:
        pass
    from acestep.pipeline_ace_step import ACEStepPipeline
    pipe = ACEStepPipeline(checkpoint_dir=None, device_id=0, dtype="bfloat16", torch_compile=False,
                           cpu_offload=True, quantized=False, overlapped_decode=True)
    os.makedirs(SOUND, exist_ok=True)
    for t in range(1, a.takes + 1):
        out = os.path.join(SOUND, f"music_t{t}.wav")
        res = pipe(prompt=a.prompt, lyrics="[inst]", audio_duration=a.seconds, infer_step=a.steps,
                   guidance_scale=15.0, scheduler_type="euler", cfg_type="apg", omega_scale=10.0,
                   manual_seeds=[a.seed + t], format="wav", save_path=out)
        print(f"take {t}: {res[0] if isinstance(res, list) else out}")


if __name__ == "__main__":
    main()
