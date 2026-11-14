"""Check Blake's WebSocket bridge without a browser.

    # terminal 1 (from water_backend/, so the venv and the models are found)
    .venv/bin/python ../Blake_frontend/server/ws_bridge.py
    .venv/bin/python ../Blake_frontend/server/ws_bridge.py --source examples/hello.mp4

    # terminal 2
    .venv/bin/python ../Blake_frontend/server/ws_probe.py

Prints the message rate, a few state snapshots, the preview frame size, and exercises the
command channel (a 2 s word capture) so a broken handler shows up here instead of during the
demo. Exit code 1 if no state or no preview arrived.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://localhost:8765")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--show", type=int, default=3, help="how many states to pretty-print")
    a = p.parse_args()

    states: list[dict] = []
    frames = 0
    bytes_total = 0

    async with websockets.connect(a.url, max_size=None) as ws:
        print(f"connected to {a.url}")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + a.seconds
        sent_capture = False
        while loop.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            except asyncio.TimeoutError:
                print("!! 3 s with no message; is the camera/model loop alive?")
                break
            if isinstance(msg, bytes):
                frames += 1
                bytes_total += len(msg)
                if not sent_capture:
                    sent_capture = True
                    await ws.send(json.dumps({"type": "cmd", "cmd": "capture"}))
                continue
            data = json.loads(msg)
            if data.get("type") != "state":
                print("message:", data)
                continue
            states.append(data)

    if not states:
        print("FAIL: no state snapshots")
        return 1

    kinds: dict[str, int] = {}
    for s in states:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    span = states[-1]["t"] - states[0]["t"]
    print(f"\nstate messages: {len(states)} over {span:.1f} s ({len(states) / max(span, 1e-9):.1f}/s)")
    print(f"preview frames: {frames} ({frames / max(span, 1e-9):.1f}/s), "
          f"mean {bytes_total / max(frames, 1) / 1024:.0f} KB/frame")
    print(f"kinds seen: {kinds}")
    print(f"sender fps: {states[-1]['fps']}, mode: {states[-1]['mode']}, source: {states[-1]['source']}")
    print(f"final text: {states[-1]['text']!r}")

    interesting = [s for s in states if s["kind"] in ("WORD", "LETTER", "AMBIGUOUS")] or states[:a.show]
    for s in interesting[:a.show]:
        print("\n--- state ---")
        print(f"  kind={s['kind']} label={s['label']!r} stable={s['stable']} mode={s['mode']}")
        print(f"  reason: {s['reason']}")
        print(f"  word:   {json.dumps(s['word'])}")
        print(f"  letter: {json.dumps(s['letter'])}")
        if s["alternatives"]:
            print(f"  alternatives: {s['alternatives']}")

    ok = frames > 0
    print("\nOK: state + preview flowing" if ok else "\nFAIL: no preview frames")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
