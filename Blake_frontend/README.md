# Blake's part — live bridge + web UI

Takes the recognition backend's real output and ships it to a browser: one WebSocket server,
one page. Nothing in `water_backend/` or `joa_frontend/` is modified — the bridge only
*imports* the backend package, so teammates can keep editing theirs.

```
Blake_frontend/
├── index.html          live page (preview + routed decision + top-3 + sentence)
├── app.js              WS client: text = JSON state, binary = JPEG preview
├── style.css           Rev 3 palette (lilac / teal / blue / pink)
└── server/
    ├── ws_bridge.py    ws://localhost:8765 — camera -> both models -> router -> JSON/JPEG
    └── ws_probe.py     headless check: message rates, sample states, commit channel
```

## Run

```bash
# terminal 1 — the bridge (needs the backend venv: mediapipe + onnxruntime live there)
cd Signify/water_backend
.venv/bin/python ../Blake_frontend/server/ws_bridge.py                    # camera 0
.venv/bin/python ../Blake_frontend/server/ws_bridge.py --source examples/hello.mp4   # no camera
.venv/bin/python ../Blake_frontend/server/ws_bridge.py --host 0.0.0.0     # LAN demo

# terminal 2 — the page
cd Signify/Blake_frontend && python3 -m http.server 8080
# open http://localhost:8080
# bridge elsewhere? http://localhost:8080/?ws=ws://192.168.1.20:8765

# no browser:
.venv/bin/python ../Blake_frontend/server/ws_probe.py
```

`water_backend/.venv` is Python 3.11 (mediapipe 0.10.21 has no 3.13 wheel — `1.x` removed the
legacy Holistic graph the word model was trained with). Rebuild:
`uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

## Protocol

| direction | payload |
|---|---|
| bridge → page, text ~15/s | `{type:"state", mode, kind, label, stable, reason, alternatives, word, letter, text, commits, capture, fps, source}` |
| bridge → page, binary ~14/s | JPEG preview, 480 px wide, mirrored |
| page → bridge, text | `{type:"cmd", cmd:"mode\|capture\|space\|confirm\|word\|letter\|repeat\|backspace\|clear"}` |

`kind` ∈ WORD / LETTER / AMBIGUOUS / NOT_SURE / NO_HANDS; `label` is null for anything but
WORD / LETTER. `word` / `letter` each carry `label, score, top[3], fresh, hands, ok, stable,
age_s, note` — in **their own units**, never comparable (word = calibrated, letter = raw
softmax).

## Design decisions (worth saying out loud in the demo)

1. **Python owns the camera, not the page.** The page never calls `getUserMedia()`; it
   renders the exact frames the models saw. Two openers = the device gets stolen mid-demo.
2. **The bridge is the single source of truth for text.** The commit rules (calibrated 0.29
   threshold, 2 agreeing windows, cooldown, duplicate guard) live in
   `recognition_router.auto_commit` + `TextBuffer`. An earlier version of `app.js` waited for
   5 frames ≥ 0.8 — wrong unit (0.8 is a raw-softmax number; the same hello scores
   0.42 / 0.63 / 0.97 calibrated) and a re-implementation of rules that already exist. Now
   the page only displays `stable` and `text`.
3. **AUTO never picks between the two models.** `hello` with a flat hand *is* the B
   handshape, so both models fire; the router shows AMBIGUOUS and offers W / L. Switching to
   WORD or SPELL with TAB is the honest demo path.
4. **TTS stays in the browser** (`speechSynthesis`) instead of piping macOS `say` through the
   socket: no audio stream, no extra latency, works on any presenter machine.

## Measured (M2, CPU, `examples/hello.mp4` replayed as a fake camera)

| | |
|---|---|
| state messages | 15.1 /s |
| preview frames | 13.9 /s, ~30 KB each |
| first `WORD hello` after connect | ~1.5 s |
| auto-committed text | `'HELLO'` |
| backend test suite | still 75 passed (nothing in `water_backend/` was touched) |

The camera loop runs below the clip's 30 fps because MediaPipe Holistic saturates the CPU —
same trade-off the backend README documents, not a regression.

## Known limits

- Single presenter: every tab shares one camera, one mode, one text buffer.
- Preview is ~14 fps, not 30 (see above).
- If the camera can't be opened, the bridge sends `{type:"error"}` and the page shows it;
  macOS needs camera permission for whichever app launched Python.
- The old `2026hackumass/server.py` mock also listens on 8765 and emits random labels —
  stop it before a demo, or you'll be watching fake data.
