"""Mix narration, music, ambience and effects from sound/cues.csv onto a video.

cues.csv columns: file,start,gain_db,role[,norm][,fade]   (role: vo | music | amb | sfx). Files live in sound/.
  start is placed to the exact sample (48 kHz), not rounded to the millisecond.
  norm (optional): how the cue's level is measured before it is set to the role reference.
    mean  (default) mean level over the whole file. Fine for continuous sound (voice, music, beds).
    gated integrated loudness with EBU R128 gating, which ignores silence. Use it for sparse sync tracks
          (a foley.py or move.py render with a few hits in seconds of silence), otherwise their low average
          makes the mixer push them far too loud.
    none  no measurement; only gain_db is applied.
  fade (optional): the mixer's own fade-in in seconds on this cue (default 0.3 for music/amb/sfx, 0 for vo).
    Use 0 for stems that carry their own fades (every foley.py / move.py / per-shot bed render): a second fade on top
    of the stem's crossfade dips the dissolve, and on a sync stem it can swallow a hit near its start.
Music and ambience are ducked under narration with a sidechain compressor. Output is loudness-normalised to -14 LUFS.

Optional (defaults reproduce the earlier mixes exactly):
  --music-duck THRESHOLD,RATIO[,LO_HZ,HI_HZ]
                                give the music its own, deeper duck (a second sidechain compressor keyed from the same
                                VO bus, e.g. 0.02,8). With LO_HZ,HI_HZ (e.g. 0.015,10,250,4000) only that band of the
                                music (the narration's intelligibility band, split with acrossover, which sums flat) gets
                                the deeper duck; its lows and highs keep the shared duck, so the music keeps its weight
                                while the words get room. The ambience always keeps the shared duck (threshold 0.05, ratio 4).
  --tp DBTP                     loudnorm true-peak ceiling (default -1). AAC encoding adds up to about 1 dB of
                                overshoot, so use -2 when the delivered file must stay under -1 dBTP.

Usage: python tools/mix.py --video output/roughcut.mp4 --out output/mixed.mp4
       [--cues sound/cues.csv] [--audio-only sound/mix.wav] [--music-duck 0.02,8] [--tp -2]
Missing cue files are skipped with a warning, so a partial mix works while files are still being generated.
"""
import argparse, csv, os, subprocess
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(HERE, "sound")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cues", default=os.path.join(SOUND, "cues.csv"))
    ap.add_argument("--audio-only", default=None, help="also write the mixed audio alone to this wav")
    ap.add_argument("--music-out", type=float, default=47.5, help="second at which the music starts its 3 s fade out")
    ap.add_argument("--music-duck", default=None, metavar="THRESHOLD,RATIO[,LO_HZ,HI_HZ]",
                    help="duck the music (or only its LO_HZ-HI_HZ band) with its own sidechain compressor, e.g. 0.02,8 "
                         "or 0.015,10,250,4000; default: one shared duck")
    ap.add_argument("--tp", type=float, default=-1.0, help="loudnorm true-peak ceiling in dBTP (default -1)")
    a = ap.parse_args()

    cues = []
    with open(a.cues, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = os.path.join(SOUND, r["file"])
            if not os.path.exists(p):
                print(f"skip (missing): {r['file']}")
                continue
            fd = (r.get("fade") or "").strip()
            cues.append((p, float(r["start"]), float(r["gain_db"]), r["role"].strip(), (r.get("norm") or "mean").strip(),
                         float(fd) if fd else None))
    if not cues:
        raise SystemExit("no cue files found in sound/")

    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", a.video]
    for p, *_ in cues:
        cmd += ["-i", p]

    # measure each cue's mean level so generated files with wildly different loudness sit at a common reference
    def mean_db(path):
        out = subprocess.run([FFMPEG, "-i", path, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
        for line in out.splitlines():
            if "mean_volume:" in line:
                return float(line.split("mean_volume:")[1].split("dB")[0])
        return -20.0

    def gated_lufs(path):
        out = subprocess.run([FFMPEG, "-i", path, "-af", "ebur128=framelog=quiet", "-f", "null", "-"], capture_output=True, text=True).stderr
        val = None
        for line in out.splitlines():
            if line.strip().startswith("I:") and "LUFS" in line:
                val = float(line.split("I:")[1].split("LUFS")[0])
        return val if val is not None and val > -70 else mean_db(path)

    REF = {"vo": -20.0, "music": -22.0, "amb": -24.0, "sfx": -22.0}  # target mean dBFS per role before the cue gain
    fc, buses = [], {"vo": [], "music": [], "amb": [], "sfx": []}
    for i, (p, start, gain, role, mode, fade_in) in enumerate(cues, start=1):
        n = int(round(start * 48000))  # sample-exact delay (int(start*1000) used to truncate, e.g. 4.6 s -> 4599 ms)
        norm = 0.0 if mode == "none" else REF[role] - (gated_lufs(p) if mode == "gated" else mean_db(p))
        if fade_in is None:
            fade_in = 0.0 if role == "vo" else 0.3
        fade = f",afade=t=in:d={fade_in:g}" if fade_in > 0 else ""
        if role == "music":
            fade += f",afade=t=out:st={a.music_out:.1f}:d=3"  # music out under the last line, before the title
        fc.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={norm + gain:.1f}dB{fade},adelay={n}S|{n}S[c{i}]")
        buses[role].append(f"[c{i}]")

    def mixbus(name, inputs):
        if not inputs:
            fc.append(f"anullsrc=r=48000:cl=stereo,atrim=0:0.1[{name}]")
        elif len(inputs) == 1:
            fc.append(f"{inputs[0]}acopy[{name}]")
        else:
            fc.append(f"{''.join(inputs)}amix=inputs={len(inputs)}:normalize=0:dropout_transition=0[{name}]")

    mixbus("vo", buses["vo"])
    mixbus("music", buses["music"])
    mixbus("amb", buses["amb"])
    mixbus("sfx", buses["sfx"])
    duck = "attack=20:release=400:makeup=1"
    if a.music_duck:
        # music and ambience ducked separately, keyed from the same VO bus (the gain depends only on the key, so the
        # ambience duck is identical to the shared one); the music gets a lower threshold / higher ratio
        md = [float(v) for v in a.music_duck.split(",")]
        if len(md) not in (2, 4):
            raise SystemExit("--music-duck takes THRESHOLD,RATIO or THRESHOLD,RATIO,LO_HZ,HI_HZ")
        m_thr, m_ratio = md[0], md[1]
        fc.append("[vo]asplit=3[vo_main][vo_sc][vo_scm]")
        fc.append(f"[amb][vo_sc]sidechaincompress=threshold=0.05:ratio=4:{duck}[amb_ducked]")
        if len(md) == 4:
            # band-limited: the music's LO-HI band gets the deep duck, the rest of it the shared one
            fc.append(f"[music]acrossover=split={md[2]:g} {md[3]:g}[m_lo][m_mid][m_hi]")
            fc.append("[vo_scm]asplit=2[vo_scm1][vo_scm2]")
            fc.append("[m_lo][m_hi]amix=inputs=2:normalize=0:dropout_transition=0[m_rest]")
            fc.append(f"[m_rest][vo_scm1]sidechaincompress=threshold=0.05:ratio=4:{duck}[m_rest_d]")
            fc.append(f"[m_mid][vo_scm2]sidechaincompress=threshold={m_thr:g}:ratio={m_ratio:g}:{duck}[m_mid_d]")
            fc.append("[m_rest_d][m_mid_d]amix=inputs=2:normalize=0:dropout_transition=0[music_ducked]")
        else:
            fc.append(f"[music][vo_scm]sidechaincompress=threshold={m_thr:g}:ratio={m_ratio:g}:{duck}[music_ducked]")
        fc.append("[music_ducked][amb_ducked]amix=inputs=2:normalize=0:dropout_transition=0[bed_ducked]")
    else:
        # bed = music + ambience, ducked 6 dB under narration
        fc.append("[music][amb]amix=inputs=2:normalize=0:dropout_transition=0[bed]")
        fc.append("[vo]asplit=2[vo_main][vo_sc]")
        fc.append(f"[bed][vo_sc]sidechaincompress=threshold=0.05:ratio=4:{duck}[bed_ducked]")
    fc.append("[vo_main]acompressor=threshold=-18dB:ratio=3:attack=5:release=80:makeup=3[vo_c]")
    fc.append(f"[vo_c][bed_ducked][sfx]amix=inputs=3:normalize=0:dropout_transition=0,loudnorm=I=-14:TP={a.tp:g}:LRA=9[aout]")

    cmd += ["-filter_complex", ";".join(fc), "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", a.out]
    subprocess.run(cmd, check=True)
    print(f"wrote {a.out}  ({len(cues)} cues)")
    if a.audio_only:
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", a.out, "-vn", "-c:a", "pcm_s16le", a.audio_only], check=True)
        print(f"wrote {a.audio_only}")


if __name__ == "__main__":
    main()
