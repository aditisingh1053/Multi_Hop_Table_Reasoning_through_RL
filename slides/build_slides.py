"""Build local copies of the presentation from the deck source in slides/source/.

  python slides/build_slides.py html   -> slides/slides.html (standalone; open in a browser)
  (Chrome/Edge headless prints slides.html to slides/slides.pdf; see slides/README.md)
  python slides/build_slides.py pptx   -> slides/slides.pptx (one full-slide image per page + speaker notes)

The PPTX step needs python-pptx and PyMuPDF (installed into tools/pylib for this project).
"""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "source"

CSS = """
@page { size: 1920px 1080px; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #5f6670; }
section { width: 1920px; height: 1080px; position: relative; overflow: hidden;
          break-after: page; page-break-after: always; margin: 0 auto 24px auto; }
h1, h2, h3, p, ul, ol { margin: 0; }
ul, ol { padding-left: 1.25em; }
li + li { margin-top: 4px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 0.35em 0.6em; border-bottom: 1px solid #D5DAE3; text-align: left; vertical-align: top; }
th { font-weight: 600; }
aside { display: none; }
@media print { html, body { background: none; } section { margin: 0; } }
"""


def arrow_svg(match):
    style = match.group(1)
    color = re.search(r"background:\s*([^;]+)", style).group(1).strip()
    w = int(re.search(r"width:\s*(\d+)px", style).group(1))
    h = int(re.search(r"height:\s*(\d+)px", style).group(1))
    head = 0.6 * w   # the deck format draws the head over the last 40% of the length
    pts = f"0,{h * 0.32:.1f} {head:.1f},{h * 0.32:.1f} {head:.1f},0 {w},{h / 2:.1f} {head:.1f},{h} {head:.1f},{h * 0.68:.1f} 0,{h * 0.68:.1f}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="flex:none" aria-label="arrow">'
            f'<polygon points="{pts}" fill="{color}"/></svg>')


def load_deck():
    deck = json.loads((SRC / "deck.json").read_text(encoding="utf-8"))
    slides = [(sid, (SRC / "slides" / f"{sid}.html").read_text(encoding="utf-8")) for sid in deck["order"]]
    return deck, slides


def build_html():
    deck, slides = load_deck()
    links = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values() if "href" in f)
    body = []
    for _, s in slides:
        s = re.sub(r'<x-shape kind="arrow-right" style="([^"]*)"></x-shape>', arrow_svg, s)
        assert "<x-" not in s, "unconverted custom element"
        body.append(s.strip())
    page = (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<title>{html.escape(deck["title"])}</title>\n{links}\n<style>{CSS}</style>\n</head>\n<body>\n'
            + "\n".join(body) + "\n</body>\n</html>\n")
    (HERE / "slides.html").write_text(page, encoding="utf-8")
    print("wrote", HERE / "slides.html")


def build_pptx():
    import pymupdf
    from pptx import Presentation
    from pptx.util import Emu

    deck, slides = load_deck()
    pdf = pymupdf.open(HERE / "slides.pdf")
    assert pdf.page_count == len(slides), f"PDF has {pdf.page_count} pages, deck has {len(slides)} slides"
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 16:9, 13.333 x 7.5 in
    img_dir = HERE / "png"
    img_dir.mkdir(exist_ok=True)
    for i, (sid, src) in enumerate(slides):
        page = pdf[i]
        zoom = 1920 / page.rect.width
        img = img_dir / f"{i + 1:02d}_{sid}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(img)
        slide = prs.slides.add_slide(prs.slide_layouts[6])               # blank layout
        slide.shapes.add_picture(str(img), 0, 0, width=prs.slide_width, height=prs.slide_height)
        notes = re.search(r"<aside>(.*?)</aside>", src, re.S)
        if notes:
            slide.notes_slide.notes_text_frame.text = html.unescape(notes.group(1).strip())
    prs.save(HERE / "slides.pptx")
    print("wrote", HERE / "slides.pptx", "and", img_dir)


if __name__ == "__main__":
    {"html": build_html, "pptx": build_pptx}[sys.argv[1]]()
