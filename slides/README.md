# Presentation slides

Nine slides (title + Problem & Motivation, Related Work, Dataset & Approach, Progress So Far, RiL-PPO Method, RiL-PPO Results, Ablations, Takeaways & Next Steps).

| File | Use |
|---|---|
| `slides.pdf` | Present or submit as is (16:9, one slide per page) |
| `slides.pptx` | PowerPoint version: each slide is a full-slide image; the speaker notes are in the notes pane |
| `slides.html` | Standalone version; open in any browser (fonts load from Google Fonts) |
| `png/` | Each slide as a 1920×1080 image |
| `source/` | The deck's source files (`deck.json` and one HTML file per slide) |
| `build_slides.py` | Rebuilds the files above from `source/` |

The PPTX slides are images, so their text cannot be edited in PowerPoint. For an editable PowerPoint file, open the online deck and download it as PPTX from there. Alternatively, edit `source/slides/*.html` and rebuild:

```bash
python slides/build_slides.py html
"C:/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-pdf-header-footer \
    --virtual-time-budget=15000 --print-to-pdf=slides/slides.pdf slides/slides.html
PYTHONPATH=tools/pylib python slides/build_slides.py pptx
```

The title slide carries the authors, course, TA, department, institute and date.
