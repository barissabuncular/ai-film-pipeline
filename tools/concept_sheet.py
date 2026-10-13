"""Contact sheets for concept-image rounds: one sheet per subject (rows = looks, columns = candidates) and an
overview of picks.

Folder layout it reads (what run_shots.py --out-dir writes): <round>/<SUBJECT>_<LOOK>/kf_<seed>.png, e.g.
concepts/diablo/r1/C09_B/kf_7520.png. Several rounds can be combined; later rounds get their own row labelled r2 etc.

Usage:
  python tools/concept_sheet.py --rounds concepts/diablo/r1 [concepts/diablo/r2] --out concepts/diablo/sheets
         [--names concepts/diablo/subject_names.csv] [--picks concepts/diablo/picks.csv] [--thumb 420]
  names: id,name (e.g. C09,Deckard Cain)   picks: subject,file,note (the chosen image per subject, starred and
  collected into overview_picks_N.jpg)
"""
import argparse, csv, os, re
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/arial.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def collect(rounds):
    """subject -> list of (row_label, [paths])"""
    subj = {}
    for rdir in rounds:
        rname = os.path.basename(os.path.normpath(rdir))
        for d in sorted(os.listdir(rdir)):
            m = re.match(r"^([A-Z]\d\d)_([A-Z])(?:_.*)?$", d)
            full = os.path.join(rdir, d)
            if not m or not os.path.isdir(full):
                continue
            files = sorted(os.path.join(full, f) for f in os.listdir(full) if f.endswith(".png"))
            if files:
                subj.setdefault(m.group(1), []).append((f"{rname} look {m.group(2)}", files))
    return subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--picks", default=None)
    ap.add_argument("--thumb", type=int, default=420)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = {}
    if a.names and os.path.exists(a.names):
        names = {r["id"]: r["name"] for r in csv.DictReader(open(a.names, encoding="utf-8"))}
    picks = {}
    if a.picks and os.path.exists(a.picks):
        for r in csv.DictReader(open(a.picks, encoding="utf-8")):
            picks[r["subject"]] = (os.path.normcase(os.path.abspath(r["file"])), r.get("note", ""))

    tw = a.thumb
    th = tw * 9 // 16
    lab_w, head_h, cap_h = 150, 46, 18
    f_head, f_lab, f_cap = font(26), font(17), font(13)
    subj = collect(a.rounds)
    for sid, rows in sorted(subj.items()):
        cols = max(len(files) for _, files in rows)
        W = lab_w + cols * (tw + 6) + 6
        H = head_h + len(rows) * (th + cap_h + 6) + 6
        sheet = Image.new("RGB", (W, H), (22, 22, 26))
        d = ImageDraw.Draw(sheet)
        d.text((10, 10), f"{sid}  {names.get(sid, '')}", fill=(240, 240, 240), font=f_head)
        for ri, (label, files) in enumerate(rows):
            y = head_h + ri * (th + cap_h + 6)
            d.text((10, y + th // 2 - 10), label, fill=(200, 200, 210), font=f_lab)
            for ci, p in enumerate(files):
                x = lab_w + ci * (tw + 6)
                im = Image.open(p).convert("RGB")
                im.thumbnail((tw, th))
                sheet.paste(im, (x, y))
                seed = os.path.splitext(os.path.basename(p))[0].replace("kf_", "")
                is_pick = sid in picks and picks[sid][0] == os.path.normcase(os.path.abspath(p))
                d.text((x + 2, y + th + 1), ("* PICK  " if is_pick else "") + f"seed {seed}",
                       fill=(255, 210, 0) if is_pick else (170, 170, 180), font=f_cap)
                if is_pick:
                    d.rectangle([x - 3, y - 3, x + im.width + 2, y + im.height + 2], outline=(255, 210, 0), width=4)
        sheet.save(os.path.join(a.out, f"{sid}.jpg"), quality=88)
    print(f"{len(subj)} subject sheets -> {a.out}")

    if picks:
        items = [(sid, *picks[sid]) for sid in sorted(picks)]
        per, cols = 12, 3
        for n in range(0, len(items), per):
            chunk = items[n:n + per]
            rows_n = -(-len(chunk) // cols)
            tw2, th2 = 640, 360
            sheet = Image.new("RGB", (cols * (tw2 + 8) + 8, rows_n * (th2 + 30) + 8), (22, 22, 26))
            d = ImageDraw.Draw(sheet)
            for i, (sid, path, note) in enumerate(chunk):
                x = 8 + (i % cols) * (tw2 + 8)
                y = 8 + (i // cols) * (th2 + 30)
                im = Image.open(path).convert("RGB")
                im.thumbnail((tw2, th2))
                sheet.paste(im, (x, y))
                d.text((x + 2, y + th2 + 4), f"{sid} {names.get(sid, '')}"[:70], fill=(240, 240, 240), font=f_lab)
            sheet.save(os.path.join(a.out, f"overview_picks_{n // per + 1}.jpg"), quality=88)
        print(f"overview sheets for {len(items)} picks")


if __name__ == "__main__":
    main()
