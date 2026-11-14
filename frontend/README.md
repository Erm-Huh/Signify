# Signify frontend (Rev 4)

A plain HTML/CSS/JS page with no build step. It connects the design to the SignBridge backend in `../water_backend/`.

## How it works with the backend

The backend recognizes **one isolated sign per video clip**. It knows **250 signs** and takes **a few seconds per clip**, so the UI is built around **record, then upload**:

1. Turn on the camera. Press **Record a sign** to record about 3 seconds (press again to stop early), or use **Upload a clip**.
2. The page sends the clip to `POST /predict`.
3. The top guess is shown. If `confident` is `false`, it shows **"Not sure"** and the top-3 guesses to choose from.
4. Every result is added to **Your sentence** right away. Unsure words are dashed; tap one to pick another guess.
5. **Say it** speaks the sentence with the browser's speech synthesis and adds it to the conversation.
6. **Tap to say** phrases cover things the model doesn't know (it has no *help*, *pain* or *medicine*).
7. **Signs Signify knows** lists `GET /labels`, with a dot showing how reliable each sign is on new signers.

The header shows `GET /health` status: Model ready, Model not loaded, or Server offline.

## Run it

Start the backend (see `../water_backend/README.md`):

```bash
cd water_backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Serve this folder. The camera needs `localhost` or HTTPS:

```bash
cd frontend && python3 -m http.server 5173
```

Then open http://localhost:5173.

- **Settings** sets the backend URL (default `http://localhost:8000`).
- **Demo mode** returns canned answers with no backend. Use it as a backup if the backend fails on stage.

## On GitHub Pages

Pages only hosts this static folder. The backend still has to run somewhere: on the demo laptop (the page can call `http://localhost:8000`) or on a separate host.
