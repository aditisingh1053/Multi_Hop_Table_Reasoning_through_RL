"""Project architecture diagram -> architecture.html (SVG). Render to PNG with headless Chrome:

  python slides/architecture/make_architecture.py
  "C:/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --hide-scrollbars \
      --window-size=2160,1580 --force-device-scale-factor=1.5 --virtual-time-budget=8000 \
      --screenshot=D:/7th_sem/oml/slides/architecture/architecture.png \
      file:///D:/7th_sem/oml/slides/architecture/architecture.html
"""
from html import escape
from pathlib import Path

W, H = 2160, 1580
NAVY, RED, INK, GREY = "#17253F", "#8C1D18", "#1E1E1E", "#5C6370"
out = []


def box(x, y, w, h, title, lines, kind="frozen", tag=None):
    """kind: frozen (grey fill), learned (white, thick navy), ours (white, thick red), data, script."""
    fill = {"frozen": "#EEF0F3", "learned": "#FFFFFF", "ours": "#FFFFFF", "data": "#FFFFFF",
            "script": "#FFFFFF"}[kind]
    stroke = {"frozen": "#8A919C", "learned": NAVY, "ours": RED, "data": "#8A919C", "script": "#8A919C"}[kind]
    sw = 3.5 if kind in ("learned", "ours") else 2
    dash = ' stroke-dasharray="9,6"' if kind == "data" else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="{sw}"{dash}/>')
    ty = y + 38
    col = RED if kind == "ours" else NAVY
    out.append(f'<text x="{x + 18}" y="{ty}" font-size="25" font-weight="700" fill="{col}">{escape(title)}</text>')
    if tag:
        out.append(f'<text x="{x + w - 14}" y="{ty}" font-size="18" text-anchor="end" fill="{GREY}">{escape(tag)}</text>')
    for i, ln in enumerate(lines):
        red = ln.startswith("!")
        ln = ln.lstrip("!")
        out.append(f'<text x="{x + 18}" y="{ty + 34 + 29 * i}" font-size="20.5" fill="{RED if red else INK}"'
                   f'{" font-weight=\"600\"" if red else ""}>{escape(ln)}</text>')


def arrow(points, label=None, lx=None, ly=None, color="#333333", dashed=False, anchor="middle", w=2.8):
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    mk = "arR" if color == RED else "ar"
    da = ' stroke-dasharray="10,7"' if dashed else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{da} marker-end="url(#{mk})"/>')
    if label:
        out.append(f'<text x="{lx}" y="{ly}" font-size="19" text-anchor="{anchor}" fill="{color}"'
                   f'{" font-weight=\"600\"" if color == RED else ""}>{escape(label)}</text>')


def lane(y, h, label, sub):
    out.append(f'<rect x="30" y="{y}" width="{W - 60}" height="{h}" fill="#F8F9FB" stroke="#D5DAE1" stroke-width="1.5"/>')
    out.append(f'<rect x="30" y="{y}" width="110" height="{h}" fill="#E9ECF1" stroke="#D5DAE1" stroke-width="1.5"/>')
    cy = y + h / 2
    out.append(f'<text transform="translate(78,{cy}) rotate(-90)" text-anchor="middle" font-size="25" '
               f'font-weight="700" fill="{NAVY}">{escape(label)}</text>')
    out.append(f'<text transform="translate(110,{cy}) rotate(-90)" text-anchor="middle" font-size="18" '
               f'fill="{GREY}">{escape(sub)}</text>')


# ---------------- title ----------------
out.append(f'<text x="30" y="52" font-size="38" font-weight="700" font-family="&apos;Source Serif 4&apos;, Georgia, serif" '
           f'fill="{NAVY}">Reader-in-the-Loop PPO for multi-hop table + text QA: system architecture</text>')

# ---------------- lanes ----------------
lane(80, 310, "A · Data preparation", "offline, run once")
lane(420, 690, "B · Training", "RL loop, 1,024 episodes / iteration")
lane(1140, 330, "C · Inference & evaluation", "greedy policy, no training")

# ---------------- lane A ----------------
yA, hA = 130, 220
box(170, yA, 300, hA, "HybridQA dataset", ["questions + gold answers", "WikiTables-WithLinks:", "Wikipedia tables and the", "passages their cells link to"], "data", "GitHub")
box(540, yA, 300, hA, "extract_wikitables.py", ["reads the tables and", "linked passages", "→ wikitables.jsonl"], "script")
box(910, yA, 400, hA, "build_instances.py · data.py", ["one entity graph per question:", "• nodes: table rows (\"Col is val ; …\")", "  and linked passages (~49 / question)", "• edges: row → linked passages", "• gold paths by answer string match"], "script")
box(1380, yA, 330, hA, "Frozen sentence encoder", ["all-MiniLM-L6-v2 (encoder.py)", "384-d embedding of the", "question and every entity", "computed once"], "frozen")
box(1780, yA, 310, hA, "Prepared splits (.pkl)", ["train   5,000 questions", "val     1,000 (unseen tables)", "test    3,466 (official dev)", "graphs + embeddings"], "data")
for x1, x2 in [(470, 540), (840, 910), (1310, 1380), (1710, 1780)]:
    arrow([(x1, yA + hA / 2), (x2 - 4, yA + hA / 2)])

# ---------------- lane B ----------------
r1, r2, hB = 510, 840, 220
# row 1 (flow right -> left)
box(1700, r1, 390, hB, "Environment / MDP (env.py)", ["state s = (question, selected set,", "  last entity); ≤ 3 hops", "candidates ≤ 48 + STOP", "14 base features per candidate", "!+ 11 bridge features, 9 state scalars"], "script")
box(1210, r1, 400, hB, "Policy πθ (policy.py)", ["pointer-network actor, 0.97 M", "state MLP + candidate MLP (256)", "one score per candidate + STOP", "!COMMIT head: 5 · reader score", "!  + learned correction"], "learned")
box(700, r1, 380, hB, "Batched rollouts (rollout.py)", ["1,024 episodes in lock-step", "sample aₜ ~ πθ(· | sₜ)", "store sₜ, aₜ, log πθ", "STOP → COMMIT → answer ŷ"], "script")
box(170, r1, 390, hB, "Reward", ["F1(ŷ, gold answer)", "− 0.02 per entity read", "+ potential shaping κ·recall", "(telescopes to 0: optimal", " policy unchanged)"], "script")
# row 2
box(170, r2, 390, hB, "Validation & checkpoint", ["every 5 iterations:", "greedy F1 on 1,000 val questions", "keep best → actor_best.pt", "40 / 100 iterations · 3 seeds"], "script")
box(700, r2, 380, hB, "Critic Vφ + GAE", ["separate value network", "advantages Âₜ (λ = 0.95)", "return targets Ĝₜ", "(rl/algos.py)"], "learned")
box(1210, r2, 400, hB, "PPO update (rl/algos.py)", ["clipped ratio, ε = 0.2", "entropy bonus 0.01", "4 epochs × minibatch 256, Adam", "early stop if KL > 0.03"], "script")
box(1700, r2, 390, hB, "Frozen reader R (reader.py)", ["MiniLM fine-tuned on SQuAD 2.0", "reads one entity → span + score", "ReaderCache: every (question,", "entity) read stored on disk"], "frozen")

# data into env
arrow([(1935, yA + hA), (1935, r1 - 4)], "graphs + embeddings", 1950, 470, anchor="start")
# row 1 flow
arrow([(1700, r1 + 80), (1614, r1 + 80)], "state sₜ", 1657, r1 + 66)
arrow([(1210, r1 + 80), (1084, r1 + 80)], "πθ(a | s)", 1147, r1 + 66)
arrow([(700, r1 + 110), (564, r1 + 110)], "episodes", 632, r1 + 96)
# rollout steps the env
arrow([(890, r1), (890, 470), (1800, 470), (1800, r1 - 4)], "env.step(aₜ) for every episode", 1345, 460)
# env <-> reader
arrow([(1790, r1 + hB), (1790, r2 - 4)], "selected entity", 1778, 800, anchor="end")
arrow([(2010, r2), (2010, r1 + hB + 4)], "bridge span", 1900, 790, color="#8C1D18", w=3.5)
out.append('<text x="1900" y="814" font-size="19" text-anchor="middle" fill="#8C1D18" font-weight="600">(ours)</text>')
# reward -> critic
arrow([(365, r1 + hB), (365, 785), (890, 785), (890, r2 - 4)], "rewards rₜ", 600, 775)
# critic -> ppo
arrow([(1080, r2 + 110), (1206, r2 + 110)], "Âₜ, Ĝₜ", 1143, r2 + 96)
# ppo -> policy
arrow([(1410, r2), (1410, r1 + hB + 4)], "update θ, φ", 1424, 800, anchor="start")
# ppo -> checkpoint
arrow([(1410, r2 + hB), (1410, 1085), (365, 1085), (365, r2 + hB + 4)], "next iteration / evaluate", 890, 1077)

# ---------------- lane C ----------------
yC, hC = 1190, 230
box(170, yC, 300, hC, "Test question", ["question + its entity", "graph and embeddings", "(e.g. dev #2: “position 4", "… what nationality?”)"], "data")
box(540, yC, 380, hC, "Trained policy (greedy)", ["actor_best.pt", "aₜ = argmax πθ(a | sₜ)", "up to 3 hops, then STOP"], "learned")
box(990, yC, 330, hC, "Frozen reader R", ["reads each chosen entity", "!span fed back (bridge)", "live on GPU or from cache"], "frozen")
box(1390, yC, 300, hC, "COMMIT (ours)", ["chooses which selected", "entity's span is the", "final answer ŷ"], "ours")
box(1760, yC, 330, hC, "Outputs", ["scripts/eval/demo.py: hop-by-hop", "  trace of one question", "scripts/eval/evaluate.py: EM / F1", "  on 3,466 questions", "loop_report.py: tables, tests"], "script")
arrow([(470, yC + 115), (536, yC + 115)])
arrow([(920, yC + 85), (986, yC + 85)], "entity", 953, yC + 72)
arrow([(990, yC + 150), (924, yC + 150)], "span", 957, yC + 178, color="#8C1D18")
arrow([(1320, yC + 115), (1386, yC + 115)])
arrow([(1690, yC + 115), (1756, yC + 115)], "ŷ", 1723, yC + 102)
# checkpoint -> inference
arrow([(250, r2 + hB), (250, 1150), (730, 1150), (730, yC - 4)], "actor_best.pt", 490, 1140)

# ---------------- legend ----------------
ly = 1515
items = [("learned", "learned (trained by PPO)"), ("frozen", "frozen model (never trained)"),
         ("ours", "this project's contribution"), ("script", "code / computation"), ("data", "data")]
x = 170
for kind, lab in items:
    fill = "#EEF0F3" if kind == "frozen" else "#FFFFFF"
    stroke = {"learned": NAVY, "ours": RED}.get(kind, "#8A919C")
    sw = 3.5 if kind in ("learned", "ours") else 2
    dash = ' stroke-dasharray="7,5"' if kind == "data" else ""
    out.append(f'<rect x="{x}" y="{ly}" width="46" height="30" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
    out.append(f'<text x="{x + 58}" y="{ly + 22}" font-size="20" fill="{INK}">{escape(lab)}</text>')
    x += 58 + len(lab) * 10 + 40
out.append(f'<line x1="{x}" y1="{ly + 15}" x2="{x + 50}" y2="{ly + 15}" stroke="{RED}" stroke-width="3.5" marker-end="url(#arR)"/>')
out.append(f'<text x="{x + 62}" y="{ly + 22}" font-size="20" fill="{INK}">reader output fed back into the state (ours)</text>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       'font-family="&apos;Source Sans 3&apos;, Arial, sans-serif">'
       '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#333333"/></marker>'
       '<marker id="arR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#8C1D18"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "\n".join(out) + "</svg>")
page = ('<!DOCTYPE html><html><head><meta charset="utf-8"><title>Architecture</title>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700'
        '&family=Source+Serif+4:opsz,wght@8..60,700&display=swap">'
        '<style>html,body{margin:0;padding:0;background:#fff}</style></head><body>' + svg + '</body></html>')
p = Path(__file__).resolve().parent / "architecture.html"
p.write_text(page, encoding="utf-8")
(Path(__file__).resolve().parent / "architecture.svg").write_text(svg, encoding="utf-8")
print("wrote", p)
