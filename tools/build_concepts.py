"""Build a concept-render CSV from a visual bible (markdown).

The bible holds three kinds of machine-readable lines:
  - Block `NAME`: reusable description (pasted word for word wherever {NAME} appears)
  - Shot `ID` (ext|int|child): shot text, may contain {NAME} placeholders
  - Look `KEY`: style block. Keys: A, B_ext, B_int, B_child, C_ext, C_int
Each shot is rendered once per look (A, B, C); B and C use the shot's ext/int/child variant.

Usage (from the project root):
  python tools/build_concepts.py story/diablo/look/visual_bible.md concepts/diablo/concepts_r1.csv
Options: --seed-base 7000 --seed-step 10 --min-words 60 --max-words 110 --width 1280 --height 720
Fails (exit 1) if a prompt is out of the word range, a placeholder is unknown, or a banned word appears.
"""
import argparse, csv, re, sys

LINE = re.compile(r"^- (Block|Shot|Look) `([A-Za-z0-9_]+)`(?: \((ext|int|child)\))?: (.+?)\s*$")
PLACEHOLDER = re.compile(r"\{([A-Z0-9_]+)\}")
# Z-Image at cfg 1 ignores negatives and turns any named thing into picture content;
# text-bearing objects come out as gibberish; names do nothing.
BANNED = [r"\bno\b", r"\bwithout\b", r"\bnot\b", r"\bnever\b", r"\bsigns?\b", r"\btext\b", r"\bletters?\b",
          r"\bwords?\b", r"\binscriptions?\b", r"\blogos?\b", r"\bdiablo\b", r"\btristram\b", r"\bblizzard\b",
          r"\bleoric\b", r"\baidan\b", r"\bmoreina\b", r"\bjazreth\b", r"\bcain\b"]


def look_key(look, env):
    if look == "A":
        return "A"
    if look == "B":
        return {"ext": "B_ext", "int": "B_int", "child": "B_child"}[env]
    return {"ext": "C_ext", "int": "C_int", "child": "C_ext"}[env]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bible"); ap.add_argument("out_csv")
    ap.add_argument("--seed-base", type=int, default=7000); ap.add_argument("--seed-step", type=int, default=10)
    ap.add_argument("--min-words", type=int, default=60); ap.add_argument("--max-words", type=int, default=110)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--looks", default="A,B,C")
    a = ap.parse_args()

    blocks, looks, shots = {}, {}, []
    with open(a.bible, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line.rstrip("\n"))
            if not m:
                continue
            kind, name, env, text = m.groups()
            if kind == "Block":
                blocks[name] = text
            elif kind == "Look":
                looks[name] = text
            else:
                shots.append((name, env or "int", text))

    order = {"L": 0, "C": 1}
    shots.sort(key=lambda s: (order.get(s[0][0], 2), s[0]))
    errors, rows, n = [], [], 0
    for sid, env, text in shots:
        def sub(m):
            if m.group(1) not in blocks:
                errors.append(f"{sid}: unknown block {{{m.group(1)}}}")
                return m.group(0)
            return blocks[m.group(1)]
        body = PLACEHOLDER.sub(sub, text)
        for look in a.looks.split(","):
            n += 1
            key = look_key(look, env)
            prompt = f"{body}, {looks[key]}"
            words = len(prompt.replace(",", " ").split())
            rid = f"{sid}_{look}"
            if not a.min_words <= words <= a.max_words:
                errors.append(f"{rid}: {words} words (want {a.min_words}-{a.max_words})")
            for pat in BANNED:
                if re.search(pat, prompt, re.I):
                    errors.append(f"{rid}: banned pattern {pat}")
            rows.append({"id": rid, "image_prompt": prompt, "motion_prompt": "",
                         "seed": a.seed_base + n * a.seed_step, "width": a.width, "height": a.height, "_w": words})

    for r in rows:
        print(f"{r['id']:8s} {r['_w']:3d} words  seed {r['seed']}")
    if errors:
        print("\nERRORS:\n  " + "\n  ".join(errors)); sys.exit(1)
    with open(a.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_prompt", "motion_prompt", "seed", "width", "height"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} rows ({len(shots)} shots x {len(a.looks.split(','))} looks) -> {a.out_csv}")


if __name__ == "__main__":
    main()
