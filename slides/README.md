# Slides

| Folder | Contents |
|---|---|
| `final/` | Final presentation: `final_presentation.pdf`, `final_presentation.pptx` (speaker notes in the notes pane). Source: `deck.html`; figures: `fig/` |
| `progress/` | Progress-evaluation presentation: `slides.pdf`, `slides.pptx`; source in `source/` |
| `architecture/` | System architecture diagram: `architecture.png` / `.svg` |

The PPTX files contain one full-slide image per page, so edit the HTML source and rebuild:

```bash
# final deck (figures from results/loop, then PDF via Chrome, then PPTX)
python slides/final/make_figures.py
"C:/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-pdf-header-footer \
    --virtual-time-budget=20000 --print-to-pdf=slides/final/final_presentation.pdf \
    file:///D:/7th_sem/oml/slides/final/deck.html
python slides/final/build_pptx.py          # needs pymupdf and python-pptx

# architecture diagram (see the header of make_architecture.py for the PNG step)
python slides/architecture/make_architecture.py
```
