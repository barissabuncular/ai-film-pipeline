"""Generate narration lines locally with Maya1 (3B, voice designed from a text description, emotion tags).

Reads sound/vo_lines.csv (file,text) and writes sound/<file>.wav (24 kHz).
Run with the maya venv (AUDIO_ROOT\\.venv-maya):
  python tools\\gen_voice_maya.py [--description "..."] [--only vo_01] [--outdir sound/tts_test/maya]
  --description  voice description: age, accent, pitch, timbre, pacing. Inline tags in text: <sigh> <whisper> <angry> ...
  --quant        none | 8bit | 4bit   (bf16 needs ~7 GB; use 8bit/4bit if you hit OOM on 8 GB)
  --takes N      N takes per line (<file>_t1.wav ...). Default 1 -> <file>.wav
"""
import argparse, csv, os, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND = os.path.join(HERE, "sound")
# Maya1 was trained on this attribute vocabulary (see prompt.txt in the HF repo):
#   age 20s/30s/40s, accent american/british/..., pitch low/normal/high, timbre deep/warm/gravelly/raspy/throaty/harsh,
#   pacing very_slow/slow/conversational/fast, emotion neutral/sad/dry/..., intensity low/med/high,
#   creative characters: dramatic_narrator, dark_villain, mafia, spy, gangster, ...
DEFAULT_DESC = ("Creative, dramatic_narrator character. Male voice in their 40s with a american accent. "
                "Low pitch, gravelly timbre, very_slow pacing, dry tone at low intensity, tired and weary film narrator.")

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SOH_ID, EOH_ID, SOA_ID, TEXT_EOT_ID = 128259, 128260, 128261, 128009


def build_prompt(tok, description, text):
    return (tok.decode([SOH_ID]) + tok.bos_token + f'<description="{description}"> {text}' + tok.decode([TEXT_EOT_ID])
            + tok.decode([EOH_ID]) + tok.decode([SOA_ID]) + tok.decode([CODE_START_TOKEN_ID]))


def unpack(ids):
    try:
        ids = ids[:ids.index(CODE_END_TOKEN_ID)]
    except ValueError:
        pass
    ids = [i for i in ids if SNAC_MIN_ID <= i <= SNAC_MAX_ID]
    n = len(ids) // 7
    l1, l2, l3 = [], [], []
    for i in range(n):
        s = [(x - CODE_TOKEN_OFFSET) % 4096 for x in ids[i * 7:(i + 1) * 7]]
        l1.append(s[0]); l2 += [s[1], s[4]]; l3 += [s[2], s[3], s[5], s[6]]
    return [l1, l2, l3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default=os.path.join(SOUND, "vo_lines.csv"))
    ap.add_argument("--outdir", default=SOUND)
    ap.add_argument("--description", default=DEFAULT_DESC)
    ap.add_argument("--model", default="maya-research/maya1")
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    import torch, soundfile as sf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC

    kw = dict(dtype=torch.bfloat16, device_map="cuda:0")
    if a.quant != "none":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if a.quant == "8bit"
                                     else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                             bnb_4bit_quant_type="nf4"))
    print("loading", a.model, a.quant)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")

    os.makedirs(a.outdir, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    with open(a.lines, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file") and (not only or r["file"] in only)]

    for r in rows:
        for t in range(1, a.takes + 1):
            torch.manual_seed(a.seed + t)
            t0 = time.time()
            inp = tok(build_prompt(tok, a.description, r["text"]), return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=2048, min_new_tokens=28, do_sample=True,
                                     temperature=a.temperature, top_p=a.top_p, repetition_penalty=a.rep,
                                     eos_token_id=CODE_END_TOKEN_ID, pad_token_id=tok.pad_token_id)
            ids = out[0, inp["input_ids"].shape[1]:].tolist()
            levels = unpack(ids)
            if len(levels[0]) == 0:
                print(r["file"], "no audio tokens generated"); continue
            codes = [torch.tensor(l, dtype=torch.long, device="cuda").unsqueeze(0) for l in levels]
            with torch.inference_mode():
                audio = snac.decoder(snac.quantizer.from_codes(codes))[0, 0].cpu().numpy()
            audio = audio[2048:] if len(audio) > 2048 else audio
            name = f"{r['file']}_t{t}.wav" if a.takes > 1 else f"{r['file']}.wav"
            outp = os.path.join(a.outdir, name)
            sf.write(outp, audio, 24000)
            print(f"{r['file']} take {t}: {len(audio)/24000:.1f}s audio in {time.time()-t0:.1f}s -> {outp}", flush=True)
    print("peak VRAM MiB:", int(torch.cuda.max_memory_allocated() / 2**20))


if __name__ == "__main__":
    main()
