#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic). Doc tooling for the README reel.
"""shoot.py <out dir>: the README reel's stills, from two demo decks (A = Ollama stopped, the TensorFold recipe row; B = Ollama up, for the
guard). Demo mode only: nothing real is touched. Stills at 3840×2160 (1920×1080 CSS px, 2×) in <out>/hi, small copies in <out>/png."""
import os, sys, time
from cdp import Browser

OUT = sys.argv[1]; A = "http://127.0.0.1:8297/"; Bu = "http://127.0.0.1:8298/"
os.makedirs(f"{OUT}/hi", exist_ok=True); os.makedirs(f"{OUT}/png", exist_ok=True)
b = Browser(os.path.abspath(f"{OUT}/prof"))
N = [0]


def snap(name):
    N[0] += 1; p = f"{OUT}/hi/{name}.png"; b.shot(p); print("shot", name, flush=True)


def until(expr, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if b.js(expr): return True
        time.sleep(0.5)
    raise RuntimeError(f"timed out: {expr}")


def top(sel=None, block="start"):
    b.js(f"document.querySelector({sel!r}).scrollIntoView({{block:{block!r}}})" if sel else "window.scrollTo(0,0)", 0.6)


QUIET = "window.confirm=()=>true;window.alert=()=>{};"
# A: Ollama stopped — sign in, the shelf, the engine filter
b.go(A, 4); b.js(QUIET)
b.js("showLogin()", 0.6); snap("01-signin")
b.js("$('login').classList.remove('on')", 0.3); top(); snap("02-library")
b.js("setEng('TensorFold')", 0.8); top(".engbar"); snap("03-engine")
# B: Ollama up — the guard
b.go(Bu, 4); b.js(QUIET)
b.js("setEng('vLLM')", 0.8)
b.js("insert(document.querySelector('tr[data-id=\"Qwen/Qwen3-32B/original\"] button.ins'),null)", 2.0)
until("$('dlg').style.display=='flex'"); b.js("document.querySelectorAll('.dk')[0].click()", 0.4); top("#dlg"); snap("04-guard")
# A again: play the TensorFold row
b.go(A, 4); b.js(QUIET); b.js("setEng('TensorFold')", 0.6)
b.js("insert(document.querySelector('tr[data-id=\"Qwen/Qwen3.8-Flash-Next/Vontra-MLX-4bit-MTP\"] button.ins'))", 0.9)
b.js("load(0)", 0.8); top(); snap("05-loading")
until("D&&D.playing&&D.playing.id&&!D.job.running"); b.js("setEng('all');load(0)", 1.5); top(); snap("06-playing")
# the long-context decode at 1 / 2 / 4 / 8 streams
b.js("BK2L=new Set([1,2,4,8]);BK2C=32768;BK2O=512;draw()", 0.3)
b.js("bk2Run('decode2')", 2.5); b.js("load(0)", 1.0); top(".lc"); snap("07-decode-running")
until("!(D.bench.job||{}).running", 240); b.js("load(0)", 1.2); top(".lc"); snap("08-decode")
b.js("bk2Run('sweep')", 1.0); until("!(D.bench.job||{}).running", 240); b.js("load(0)", 1.2)
b.js("[...document.querySelectorAll('.lc .why')].find(e=>e.textContent.startsWith('Last prefill')).scrollIntoView({block:'start'})", 0.6); snap("09-sweep")
b.js("bk2Run('quality')", 1.0); until("!(D.bench.job||{}).running", 600); b.js("load(0)", 1.2)
b.js("bk2Cmp(D.bench.v2.quality_n-2,D.bench.v2.quality_n-1)", 1.5)
b.js("[...document.querySelectorAll('.lc .why')].find(e=>e.textContent.startsWith('Quality test')).scrollIntoView({block:'center'})", 0.6); snap("10-quality")
# history and alerts
top("#hist"); b.js("histSet('24h')", 1.5); top("#hist"); snap("11-history")
b.js("histSet('30d')", 1.5); top("#hist"); snap("12-history-30d")
b.js("ntfAct('new_topic')", 1.2); b.js("$('ntfon').checked=true;ntfSave()", 1.2); top("#alerts"); snap("13-alerts")
# a phone
b.size((430, 932), 3, True); b.go(A, 4); b.js(QUIET); top(); snap("p1-phone"); top("#tape"); b.js("window.scrollBy(0,560)", 0.6); snap("p2-phone")
b.close()
print("done", N[0])
