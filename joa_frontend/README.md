# Signify website (wireframe V4)

Plain HTML/CSS/JS, no build step. Fonts come from Google Fonts (DM Sans). GitHub Pages can serve this folder as is.

| File | Page |
|---|---|
| `index.html` | Home |
| `visit.html` | Translate: camera, tap to say, speaker box (speech to text + name catching), text to speech, visit summary |
| `why.html` | Why it matters |
| `how.html` | How it works |
| `team.html` | Team (headshots in `assets/`) |
| `assets/view.js` | Small template runtime the pages use (documented at the top of the file) |

Each page has a `<template id="view">` with the design's markup and a `class Component` with its state and behavior.

## How the Translate page gets signs (hybrid: the website + the laptop)

The website itself never touches the camera. Blake's bridge (`Blake_frontend/server/ws_bridge.py`) runs on the
laptop, owns the webcam, runs the word + letter models from `water_backend/` and decides what gets added to the
sentence. The Translate page connects to it over a WebSocket and shows its camera preview, the live sign and the
sentence, and sends commands back (capture, confirm, mode, word/letter pick, undo, clear).

```
webcam -> ws_bridge.py (laptop) --ws://localhost:8765--> visit.html (localhost or GitHub Pages)
```

The page works the same whether it is opened from `localhost` or from GitHub Pages: it always connects to the
bridge address in **Settings** (default `ws://localhost:8765`).

## Run the demo on the laptop

One-time setup (Python 3.11 is required by MediaPipe 0.10.21; `uv` downloads it):

```bash
cd water_backend
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt     # macOS/Linux: .venv/bin/python
```

Every time:

```bash
# terminal 1: the bridge (camera 0)
cd water_backend
.venv/Scripts/python.exe ../Blake_frontend/server/ws_bridge.py           # macOS/Linux: .venv/bin/python
#   no camera? replay a clip:  ... ws_bridge.py --source examples/hello.mp4

# terminal 2: the website (or just open the GitHub Pages URL)
cd joa_frontend
python -m http.server 5173
```

Open http://localhost:5173/visit.html. The badge above the camera should say **Sign bridge connected**.

- **GitHub Pages + laptop:** open the Pages URL on the laptop that runs the bridge. Chrome may ask to allow
  access to devices on your local network the first time; allow it.
- **The sentence** builds up word by word in the box under the conversation ("Sign, tap or type to start").
  It is editable: click in and type over a wrong sign. **Say it** (or Enter) speaks it and posts it to the
  conversation; **Undo** drops the last word; **Start over** clears it.
- Keyboard: Space capture · Enter confirm · M mode · W / L pick word or letter · R repeat letter · Backspace undo · C clear.
  Tab is left for keyboard navigation (Blake's page uses Tab for mode; here it is M).

## Speech

Text to speech (`say()`) and the speaker's speech to text (`toggleListenLive()`) use the browser for now
(`speechSynthesis`, and `SpeechRecognition` in Chrome/Edge). They will move to ElevenLabs through endpoints in
`water_backend/`, so the API key stays on the laptop (`water_backend/.env`, which git ignores).

## Still simulated

- Waiting room "Demo: nurse calls you" and "Request an interpreter".
- Bracketed placeholders in the copy (`[NAME]`, `[DEVPOST LINK]`, `[CITE]` and similar).
