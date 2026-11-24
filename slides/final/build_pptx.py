"""Turn final_presentation.pdf into PNG previews and a PPTX (one full-slide image per page, speaker
notes from the <aside> blocks of deck.html).

Build steps (from the repository root):
  1. python slides/final/make_figures.py                       (WSL venv: figures from results/loop)
  2. Chrome prints the deck (Windows):
     "C:/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-pdf-header-footer \
         --virtual-time-budget=20000 --print-to-pdf=slides/final/final_presentation.pdf \
         file:///D:/7th_sem/oml/slides/final/deck.html
  3. python slides/final/build_pptx.py                         (WSL venv: needs pymupdf, python-pptx)
"""
import html
import re
from pathlib import Path

import pymupdf
from pptx import Presentation
from pptx.util import Emu

HERE = Path(__file__).resolve().parent


def main():
    src = (HERE / "deck.html").read_text(encoding="utf-8")
    sections = re.findall(r"<section\b.*?</section>", src, re.S)
    notes = []
    for s in sections:
        m = re.search(r"<aside>(.*?)</aside>", s, re.S)
        notes.append(html.unescape(m.group(1).strip()) if m else "")
    pdf = pymupdf.open(HERE / "final_presentation.pdf")
    assert pdf.page_count == len(sections), f"PDF {pdf.page_count} pages vs {len(sections)} sections"
    out = HERE / "png"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 16:9
    for i, page in enumerate(pdf):
        img = out / f"slide{i + 1:02d}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(1920 / page.rect.width, 1920 / page.rect.width)).save(img)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(str(img), 0, 0, width=prs.slide_width, height=prs.slide_height)
        if notes[i]:
            slide.notes_slide.notes_text_frame.text = notes[i]
    prs.save(HERE / "final_presentation.pptx")
    print(f"wrote {HERE / 'final_presentation.pptx'} ({pdf.page_count} slides) and {out}")


if __name__ == "__main__":
    main()
