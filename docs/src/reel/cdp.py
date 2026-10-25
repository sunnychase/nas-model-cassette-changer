#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic). Doc tooling for the README reel.
"""cdp.py: a tiny standard-library Chrome DevTools client (WebSocket + JSON), enough to drive a headless page and take screenshots.
Used only to make the README reel from demo mode."""
import base64, json, os, socket, struct, subprocess, time, urllib.parse, urllib.request


class WS:
    def __init__(self, url):
        u = urllib.parse.urlsplit(url); self.s = socket.create_connection((u.hostname, u.port), timeout=60)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.s.sendall(req.encode()); buf = b""
        while b"\r\n\r\n" not in buf: buf += self.s.recv(4096)
        if b" 101 " not in buf.split(b"\r\n", 1)[0]: raise RuntimeError(buf[:200])
        self.rest = buf.split(b"\r\n\r\n", 1)[1]

    def _read(self, n):
        while len(self.rest) < n:
            c = self.s.recv(1 << 20)
            if not c: raise ConnectionError("closed")
            self.rest += c
        out, self.rest = self.rest[:n], self.rest[n:]; return out

    def send(self, text):
        b = text.encode(); h = bytearray([0x81]); n = len(b)
        if n < 126: h.append(0x80 | n)
        elif n < 65536: h.append(0x80 | 126); h += struct.pack(">H", n)
        else: h.append(0x80 | 127); h += struct.pack(">Q", n)
        m = os.urandom(4); h += m
        self.s.sendall(bytes(h) + bytes(x ^ m[i % 4] for i, x in enumerate(b)))

    def recv(self):
        parts = []
        while True:
            b0, b1 = self._read(2); op, n = b0 & 0x0F, b1 & 0x7F
            if n == 126: n = struct.unpack(">H", self._read(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._read(8))[0]
            data = self._read(n)
            if op == 8: raise ConnectionError("closed by the browser")
            if op in (1, 0): parts.append(data)
            if b0 & 0x80 and op in (1, 0): return b"".join(parts).decode("utf-8", "replace")


class Browser:
    def __init__(self, profile, port=9333, size=(1920, 1080), scale=2):
        self.p = subprocess.Popen(["chromium-browser", "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
                                   f"--user-data-dir={profile}", f"--window-size={size[0]},{size[1]}", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2)); break
            except Exception: time.sleep(0.2)
        url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
        self.ws = WS(url); self.i = 0
        self.size(size, scale)

    def call(self, method, **params):
        self.i += 1; me = self.i; self.ws.send(json.dumps({"id": me, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == me:
                if "error" in m: raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def size(self, size, scale=2, mobile=False):
        self.call("Emulation.setDeviceMetricsOverride", width=size[0], height=size[1], deviceScaleFactor=scale, mobile=mobile)

    def go(self, url, wait=3.0): self.call("Page.navigate", url=url); time.sleep(wait)

    def js(self, expr, wait=0.0):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"): raise RuntimeError(f"JS: {r['exceptionDetails'].get('text')} {str(r['exceptionDetails'].get('exception', {}).get('description', ''))[:300]}")
        if wait: time.sleep(wait)
        return (r.get("result") or {}).get("value")

    def shot(self, path):
        d = self.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as f: f.write(base64.b64decode(d["data"]))

    def close(self):
        try: self.p.terminate(); self.p.wait(10)
        except Exception: self.p.kill()
