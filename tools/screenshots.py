"""Capture docs/img/*.png from the running front end.

Needs the API on :8000 and the dev server on :5173, plus `pip install websockets`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

# Routing lets the lookup view load with a worked example already scored,
# instead of shooting an empty form.
APP = ("http://localhost:5173/?view=lookup&gene=TP53"
       "&variant=R175H%20P72R%20R273H")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
WIDTH, HEIGHT = 1200, 2400

# Tabs are clicked rather than reached by URL, so the capture exercises the same
# path a user takes. Chrome's own --screenshot flag cannot click anything.
SHOTS = [
    ("Lookup", "lookup.png", "!!document.querySelector('tbody tr')"),
    ("Heatmap", "heatmap.png",
     "!!document.querySelector('canvas') && document.querySelector('canvas').width > 100"),
    ("Structure", "structure.png",
     "(()=>{const c=document.querySelector('.viewer canvas');"
     "return !!c && c.width>100 && !document.querySelector('.status');})()"),
    ("Metrics", "metrics.png",
     "(()=>{const n=document.querySelectorAll('svg,canvas').length;"
     "return n>1 && !/Loading/i.test(document.body.innerText);})()"),
]

# .app reports its layout box, which overshoots the painted content; clipping to
# it left the metrics page half black.
CLIP = ("(()=>{const a=document.querySelector('.app');if(!a)return null;"
        "let bottom=0,right=0;"
        "a.querySelectorAll('*').forEach(e=>{const r=e.getBoundingClientRect();"
        "if(r.width>0&&r.height>0){bottom=Math.max(bottom,r.bottom);"
        "right=Math.max(right,r.right);}});"
        "if(!bottom)return null;const p=a.getBoundingClientRect().left;"
        "return {x:0,y:0,width:Math.ceil(right+p),height:Math.ceil(bottom+p)};})()")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class CDP:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def send(self, method: str, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expression: str):
        r = await self.send("Runtime.evaluate", expression=expression,
                            returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


async def wait_for(cdp: CDP, expression: str, timeout: float, label: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if await cdp.js(expression):
                return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    print(f"    ! timed out waiting for {label}")
    return False


async def run(out_dir: str) -> int:
    port = free_port()
    profile = tempfile.mkdtemp(prefix="prism-shot")
    chrome = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}", f"--window-size={WIDTH},{HEIGHT}",
        "--hide-scrollbars", "--force-device-scale-factor=2",
        # 3Dmol needs WebGL and a headless runner has no GPU.
        "--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader",
        "--no-first-run", "--no-default-browser-check", "--disable-extensions",
        "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ws_url = None
    for _ in range(60):
        try:
            tabs = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/list", timeout=2).read())
            pages = [t for t in tabs if t["type"] == "page"]
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ws_url:
        print("could not attach to Chrome")
        chrome.kill()
        return 1

    async with websockets.connect(ws_url, max_size=100 * 1024 * 1024) as ws:
        cdp = CDP(ws)
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        await cdp.send("Emulation.setDeviceMetricsOverride",
                       width=WIDTH, height=HEIGHT, deviceScaleFactor=2, mobile=False)
        await cdp.send("Page.navigate", url=APP)
        await asyncio.sleep(3)

        if not await wait_for(cdp, "!!document.querySelector('nav button')", 60, "app shell"):
            chrome.kill()
            return 1

        for label, filename, ready in SHOTS:
            print(f"  -> {label}")
            clicked = await cdp.js(
                "(()=>{const b=[...document.querySelectorAll('nav button')]"
                f".find(x=>x.textContent.trim()==='{label}');"
                "if(!b)return false;b.click();return true;})()")
            if not clicked:
                print(f"    ! no tab button {label}")
                continue

            await asyncio.sleep(1.5)
            await wait_for(cdp, ready, 120, f"{label} content")
            await asyncio.sleep(3.0)

            box = await cdp.js(CLIP)
            kwargs = {"format": "png", "captureBeyondViewport": True}
            if box:
                kwargs["clip"] = {**box, "scale": 1}
            shot = await cdp.send("Page.captureScreenshot", **kwargs)

            path = os.path.join(out_dir, filename)
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
            print(f"     {path}  {os.path.getsize(path) / 1024:.0f} KB")

    chrome.kill()
    shutil.rmtree(profile, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "docs/img")))
