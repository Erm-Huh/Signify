# Signify: Rev 3 website export

A one-to-one export of the **Revision 3** design ("clinical focus, natural look") from the design canvas,
taken from the live canvas on 2026-11-14, including your Home page edits.
It's plain HTML with no build step and no dependencies except Google Fonts (DM Sans).

## Files

| File | Page | Nav label |
|---|---|---|
| `index.html` | Home | Home |
| `visit.html` | Your visit: Front desk → Waiting room → Nurse → Doctor → Checkout | Translate |
| `why.html` | Why it matters | Why it matters |
| `how.html` | How it works | How it works |
| `team.html` | Team | Team |
| `assets/view.js` | Small template runtime the pages use (about 150 lines, documented at the top) | |

Each page has:
- `<template id="view">`: the page markup exactly as designed. Inline styles hold the design (palette: lilac `#7d5ba6`, teal `#558564`, blue `#8fd9eb`, pink `#ffc0cb`, graphite `#2f2f2f`).
- `class Component`: page state and behavior. `renderVals()` returns the values the template shows (`{{name}}`), and handlers call `this.setState(...)`.

## Run it

```bash
python3 -m http.server 8080
```

Open http://localhost:8080. Opening `index.html` straight from the folder also works.
GitHub Pages can serve this folder as is.

## What is simulated, and where to integrate (`visit.html`)

All of these live inside `class Component` in `visit.html`:

| Design behavior | Simulated by | Replace with |
|---|---|---|
| Signs appear one by one as chips | `tick()`, started by `toggleRun` | Results from the sign model. Push `{ word, alt, unsure }` into `state.chips`. `alt` is the model's 2nd guess, `unsure` = low confidence (dashed chip) |
| Gloss → English sentence | `draftEnglish` in `renderVals()` (matches the demo sentence per stop) | Your gloss-to-English step |
| "Speaking: …" indicator | `say(text)` | Real text to speech, e.g. `speechSynthesis.speak(new SpeechSynthesisUtterance(text))` with the chosen voice/speed/volume |
| Staff reply arrives automatically | `patientSays()` → `setTimeout` with `this.stops[i].auto` | Real staff input (the reply box and quick replies already work) |
| Waiting-room "nurse calls you" | `callMe` | Real queue status |
| Request an interpreter | `requestInterp` | Your VRI / interpreter request |
| Demo sentences per stop | `this.stops[i].gloss / alts / english` | Remove once the model is connected |

Everything else (stops, confirm step, Undo, Start over, tap phrases, pain scale, flip, urgent alert, voice settings, visit summary, copy) already works as designed.

## Placeholders still in the copy

`[NAME]`, `[N]`, `[X]%`, `[STAT]`, `[CITE]`, `[CONFIRM]`, `[TTS ENGINE: TBD]`, `[DEVPOST LINK]`, `[PLACEHOLDER]` and similar. They are kept exactly as in the design.
