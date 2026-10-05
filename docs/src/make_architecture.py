"""Writes docs/images/architecture.svg (render to PNG with any browser; see docs/src/README)."""
import os
W, H = 1800, 1060; o = []; A = o.append
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def t(x, y, s, size=16, c="#94a3b8", anchor="start", w="400", mono=False):
    fam = ' font-family="JetBrains Mono, ui-monospace, monospace"' if mono else ""
    A(f'<text x="{x}" y="{y}" font-size="{size}" fill="{c}" text-anchor="{anchor}" font-weight="{w}"{fam}>{esc(s)}</text>')
def card(x, y, w, h, title, sub, acc):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="url(#card)" stroke="{acc}" stroke-opacity=".8" stroke-width="1.8"/>')
    A(f'<rect x="{x}" y="{y}" width="{w}" height="6" rx="3" fill="{acc}"/>'); t(x+22, y+42, title, 24, "#f8fafc", w="800"); t(x+22, y+68, sub, 14, acc, mono=True)
def item(x, y, w, head, sub, c, h=64):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#0b1220" stroke="{c}" stroke-opacity=".5"/>')
    t(x+16, y+26, head, 16, "#e2e8f0", w="700"); t(x+16, y+48, sub, 13, "#94a3b8", mono=True)
def arrow(d, c, label=None, lx=0, ly=0, dash="8 7"):
    A(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2.6" stroke-dasharray="{dash}" marker-end="url(#a{c[1:]})"/>')
    if label:
        A(f'<rect x="{lx-8}" y="{ly-18}" width="{len(label)*7.9+16}" height="26" rx="7" fill="#020617" stroke="{c}" stroke-opacity=".45"/>'); t(lx, ly, label, 13.5, c, mono=True)
CS = ["#22d3ee", "#38b6ff", "#f472b6", "#f59e0b", "#22c55e", "#94a3b8", "#ef4444"]
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif"><defs>')
A('<radialGradient id="bg" cx="50%" cy="0%" r="120%"><stop offset="0" stop-color="#10203c"/><stop offset=".6" stop-color="#060b18"/><stop offset="1" stop-color="#03060e"/></radialGradient>')
A('<linearGradient id="card" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#111a2e"/><stop offset="1" stop-color="#0a1120"/></linearGradient>')
A('<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-opacity=".35"/></pattern>')
for c in CS: A(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>')
A(f'</defs><rect width="{W}" height="{H}" fill="url(#bg)"/><rect width="{W}" height="{H}" fill="url(#grid)"/>')
t(60, 74, "NAS MODEL CASSETTE CHANGER — HOW IT FITS TOGETHER", 34, "#f8fafc", w="800")
t(62, 108, "One big, slow, cheap library. One fast GPU box with a small local disk. Models move like cassettes: copy in, play, eject.", 18, "#93c5fd")

NX, NW = 60, 470; GX, GW = 690, 540; YX, YW = 1380, 360          # three columns, 160 px gutters for the arrow labels
card(NX, 150, NW, 560, "NAS · the library", "any Linux box with disks · may sleep", "#22d3ee")
for i, (h, s_) in enumerate([("Staging folders", "Models/<category>/<org>/<model>/"), ("mcc_index.py  (systemd timer, 10 min)", "classify · size · format · quant · status"),
                             ("library/<Maker>/<Model>/<Variant>", "relative symlinks — nothing is moved"), ("library/_index/library.json", "the machine index the deck reads"),
                             ("library/_tiers/  ·  library/_kits/", "grouped by nodes needed · serving recipes"), ("ABOUT-MODEL.md + manifest.json", "per model: what it is, fit, how to run")]):
    item(NX + 22, 236 + 76 * i, NW - 44, h, s_, "#22d3ee")
card(GX, 150, GW, 560, "GPU box · the deck", "fast local NVMe · Ollama · Docker", "#38b6ff")
item(GX + 22, 236, GW - 44, "mcc-sync.timer  →  mcc_deck.py sync", "ssh, read-only · never wakes the NAS", "#38b6ff")
item(GX + 22, 312, GW - 44, "mcc_deck.py serve   (127.0.0.1:8099)", "shelf · tiers · insert · eject · log", "#38b6ff")
item(GX + 22, 388, GW - 44, "The guard", "names every model an insert would stop → 409", "#f59e0b")
tw = (GW - 44 - 32) // 3
item(GX + 22, 464, tw, "Ollama", "GGUF", "#38b6ff"); item(GX + 22 + tw + 16, 464, tw, "vLLM · SGLang", "safetensors", "#38b6ff")
item(GX + 22 + 2 * (tw + 16), 464, tw, "Recipe lanes", "e.g. TensorFold", "#f0abfc")
item(GX + 22, 540, GW - 44, "~/models/deck/<Maker>/<Model>/<Variant>", "local copies (the EJECTOR deletes them)", "#38b6ff")
item(GX + 22, 616, GW - 44, "More nodes (optional)", "listed in the fleet; tiers show what needs 2–4", "#94a3b8")
card(YX, 150, YW, 560, "You", "phone · laptop · anywhere", "#f472b6")
for i, (h, s_) in enumerate([("The web page", "search · filter · insert · eject"), ("Access token", "~/.config/mcc/token (0600)"),
                             ("Remote access", "SSH tunnel · VPN · zero-trust"), ("OpenAI-style clients", "Ollama · vLLM · SGLang · recipe")]):
    item(YX + 22, 236 + 76 * i, YW - 44, h, s_, "#f472b6")
for i, line in enumerate(["The deck binds to localhost by default.", "Put it behind something that", "authenticates before exposing it."]):
    t(YX + 22, 580 + 24 * i, line, 14, "#f9a8d4")
g1 = (NX + NW + GX) / 2; g2 = (GX + GW + YX) / 2
def lab(cx, y, s_, c):
    w = len(s_) * 7.9 + 16; A(f'<rect x="{cx - w/2}" y="{y - 18}" width="{w}" height="26" rx="7" fill="#020617" stroke="{c}" stroke-opacity=".55"/>'); t(cx, y, s_, 13.5, c, "middle", mono=True)
arrow(f"M{NX+NW} 496 C {g1} 496 {g1} 268 {GX-2} 268", "#22d3ee"); lab(g1, 388, "index sync", "#22d3ee")
arrow(f"M{NX+NW} 572 C {g1} 572 {g1} 572 {GX-2} 572", "#22c55e"); lab(g1, 610, "rsync on INSERT", "#22c55e")
arrow(f"M{YX} 268 C {g2} 268 {g2} 344 {GX+GW+2} 344", "#f472b6"); lab(g2, 236, "HTTPS + token", "#f472b6")
arrow(f"M{YX} 496 L {GX+GW+2} 496", "#94a3b8", dash="3 6"); lab(g2, 534, "API", "#94a3b8")

t(60, 778, "AN INSERT, STEP BY STEP", 20, "#f8fafc", w="800")
steps = [("Pick", "model + quant", "#38b6ff"), ("Plan", "what would stop?", "#f59e0b"), ("Confirm", "tick every name", "#ef4444"),
         ("Copy", "rsync, resumable", "#22c55e"), ("Re-check", "plan again", "#f59e0b"), ("Play", "engine or recipe", "#38b6ff"),
         ("Verify", "protected still up?", "#f59e0b"), ("Eject", "keep or delete", "#94a3b8")]
x0, w, gap = 60, 196, 18
for i, (h, s, c) in enumerate(steps):
    x = x0 + i * (w + gap)
    A(f'<rect x="{x}" y="806" width="{w}" height="104" rx="14" fill="#0b1220" stroke="{c}" stroke-width="1.8"/>')
    A(f'<circle cx="{x+30}" cy="840" r="16" fill="{c}"/>'); t(x+30, 846, str(i+1), 15, "#020617", "middle", "800")
    t(x+56, 847, h, 19, "#f8fafc", w="800"); t(x+18, 886, s, 13.5, "#94a3b8", mono=True)
    if i < len(steps) - 1: arrow(f"M{x+w+2} 858 L{x+w+gap-2} 858", "#94a3b8", dash="0")
t(60, 960, "Nothing is stopped that you did not name. A model that does not fit is refused, not squeezed in. If a protected model is", 15.5, "#cbd5e1")
t(60, 986, "pushed out anyway, the new cassette is removed again. Ejecting never touches the NAS copy.", 15.5, "#cbd5e1")
t(60, 1026, "Tip: link the NAS and the GPU box with 10 GbE (and NAS disks fast enough to fill it): a 65 GB model copies in ~1–2 min instead of ~10 on 1 GbE.", 15.5, "#86efac")
A('</svg>')
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "images", "architecture.svg"), "w") as f: f.write("\n".join(o))
print("architecture.svg written")
