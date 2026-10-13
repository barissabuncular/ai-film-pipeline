"""Video-to-audio sound effects with MMAudio (hkchengrex/MMAudio, CVPR 2025), synchronised to the video frames.

Model: mmaudio large_44k_v2 (default) - flow-matching net conditioned on CLIP (semantics, 8 fps) +
Synchformer (timing, 25 fps) + text.  Native output is 44.1 kHz mono; pass --rate 48000 to resample.
Repo:    AUDIO_ROOT\\MMAudio      (weights in <repo>\\weights and <repo>\\ext_weights, auto-downloaded)
Venv:    AUDIO_ROOT\\.venv-v2a    (torch 2.11.0+cu128)

Example (from the project root, with the v2a venv):
  python tools\\gen_v2a.py ^
      --video shots\\<id>\\selected.mp4 ^
      --prompt "brass bullet casings falling and bouncing on a wet concrete floor, slow motion" ^
      --negative "music, speech, voice" ^
      --out shots\\<id>\\v2a_casings.wav

Options: --seconds N (generate only the first N s; default = whole video), --seed N (default 42),
         --steps N (Euler steps, default 25), --cfg F (CFG strength, default 4.5), --rate 48000,
         --variant small_44k|medium_44k|large_44k|large_44k_v2|small_16k, --composite <mp4> to also
         write the video re-muxed with the new audio, --fp32 to disable bf16 (slower, more VRAM).
The wav is trimmed/padded to exactly the video duration (or --seconds).
"""
import argparse
import os
import sys
import time
from pathlib import Path

AUDIO_ROOT = os.environ.get("AUDIO_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "audio"))
REPO = Path(AUDIO_ROOT) / "MMAudio"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch

from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate, load_video, make_video, setup_eval_logging
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def video_duration_sec(path: Path) -> float:
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if s.duration is not None and s.time_base is not None:
            return float(s.duration * s.time_base)
        if c.duration is not None:
            return c.duration / 1e6
    raise RuntimeError(f"cannot determine duration of {path}")


def resolve_paths(cfg: ModelConfig) -> ModelConfig:
    """ModelConfig paths are relative to the repo root; make them absolute so cwd does not matter."""
    cfg.model_path = REPO / cfg.model_path
    cfg.vae_path = REPO / cfg.vae_path
    cfg.synchformer_ckpt = REPO / cfg.synchformer_ckpt
    if cfg.bigvgan_16k_path is not None:
        cfg.bigvgan_16k_path = REPO / cfg.bigvgan_16k_path
    return cfg


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="MMAudio video-to-audio")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="", help="negative prompt")
    ap.add_argument("--seconds", type=float, default=0.0, help="generate only the first N seconds (0 = full video)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--variant", default="large_44k_v2", choices=sorted(all_model_cfg))
    ap.add_argument("--rate", type=int, default=0, help="output sample rate (0 = model native: 44100 or 16000)")
    ap.add_argument("--composite", type=Path, default=None, help="optional mp4 with the generated audio muxed in")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    setup_eval_logging()
    video = a.video.expanduser().resolve()
    if not video.is_file():
        sys.exit(f"video not found: {video}")

    cfg = resolve_paths(all_model_cfg[a.variant])
    cfg.download_if_needed()
    seq_cfg = cfg.seq_cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if (a.fp32 or device == "cpu") else torch.bfloat16
    print(f"device={device} dtype={dtype} variant={a.variant}")

    full_dur = video_duration_sec(video)
    duration = min(a.seconds, full_dur) if a.seconds > 0 else full_dur
    print(f"video {video.name}: {full_dur:.3f} s, generating {duration:.3f} s")

    t0 = time.time()
    net: MMAudio = get_my_mmaudio(cfg.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(cfg.model_path, map_location=device, weights_only=True))
    feature_utils = FeaturesUtils(tod_vae_ckpt=cfg.vae_path,
                                  synchformer_ckpt=cfg.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=cfg.mode,
                                  bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
                                  need_vae_encoder=False).to(device, dtype).eval()
    print(f"models loaded in {time.time() - t0:.1f} s")

    rng = torch.Generator(device=device).manual_seed(a.seed)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=a.steps)

    # frames for CLIP (8 fps, 384px) and Synchformer (25 fps, 224px); all_frames only if we re-mux a video
    video_info = load_video(video, duration, load_all_frames=a.composite is not None)
    target_dur = duration  # wav length we promise (full video, or --seconds)
    duration = video_info.duration_sec  # repo truncates to whole 25 fps sync frames (e.g. 5.0625 -> 5.00 s)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    t0 = time.time()
    audios = generate(clip_frames, sync_frames, [a.prompt], negative_text=[a.negative],
                      feature_utils=feature_utils, net=net, fm=fm, rng=rng, cfg_strength=a.cfg)
    audio = audios.float().cpu()[0]  # (samples,) mono at seq_cfg.sampling_rate
    print(f"generated in {time.time() - t0:.1f} s")

    native = seq_cfg.sampling_rate
    out_rate = a.rate or native
    if out_rate != native:
        import torchaudio.functional as AF
        audio = AF.resample(audio.unsqueeze(0), native, out_rate)[0]

    # exact video length: the latent grid rounds up and load_video may truncate a partial frame,
    # so trim (or pad with silence) to the container duration
    n = int(round(target_dur * out_rate))
    if audio.shape[-1] >= n:
        audio = audio[:n]
    else:
        audio = torch.nn.functional.pad(audio, (0, n - audio.shape[-1]))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(a.out), np.clip(audio.numpy().astype(np.float32).reshape(-1), -1.0, 1.0), int(out_rate), subtype="PCM_16")
    print(f"wrote {a.out}  ({n / out_rate:.3f} s @ {out_rate} Hz, peak {audio.abs().max():.3f})")

    if a.composite is not None:
        a.composite.parent.mkdir(parents=True, exist_ok=True)
        make_video(video_info, a.composite, audios.float().cpu()[0], sampling_rate=native)
        print(f"wrote {a.composite}")

    if device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
