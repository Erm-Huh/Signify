/* Signify — live client for Blake's WebSocket bridge (server/ws_bridge.py).
 *
 * HOW THE DATA FLOWS (this is the part that matters for the writeup)
 *   Python owns the camera -> UnifiedSession (word model + alphabet model) -> router
 *   -> JSON state ~15/s + JPEG preview ~14/s  ==>  this file  ==>  DOM.
 *
 * THREE RULES, all of them deliberate:
 *  1. Do NOT call getUserMedia() here. The bridge process already opened the camera; a
 *     second opener fights for the device and the preview flickers or dies. The frames we
 *     show are the ones the models actually saw.
 *  2. Do NOT re-derive stability or append words locally. The first version of this file
 *     waited for 5 frames >= 0.8 before accepting a label — but 0.8 is a *raw softmax*
 *     number and the word model's scores are *calibrated* (the same hello came out 0.42 /
 *     0.63 / 0.97 across clips, threshold 0.29). Counting frames here duplicated the
 *     server's rules with the wrong threshold in the wrong unit. The bridge sends `stable`
 *     and the committed `text`; we display them.
 *  3. Do NOT compare the word score with the letter score. Different models, different
 *     calibration; AUTO already refuses to guess and reports AMBIGUOUS.
 *
 * Protocol: text frames = JSON state, binary frames = JPEG preview, we send {type:'cmd'}.
 * See server/ws_bridge.py for the full schema.
 */

const params = new URLSearchParams(location.search);
// Same host as the page unless ?ws= overrides it (LAN demo: ?ws=ws://192.168.1.20:8765)
const WS_URL = params.get('ws') || 'ws://localhost:8765';
const RETRY_MS = 1200;

const el = (id) => document.getElementById(id);

let ws = null;
let retryTimer = null;
let frameUrl = null;   // previous blob URL, revoked to keep memory flat
let latest = null;     // last state message

const KIND_TEXT = {
  WORD: 'WORD',
  LETTER: 'LETTER',
  AMBIGUOUS: 'AMBIGUOUS',
  NOT_SURE: 'NOT SURE',
  NO_HANDS: 'NO HANDS',
};

const KIND_CLASS = {
  WORD: 'word',
  LETTER: 'letter',
  AMBIGUOUS: 'ambiguous',
  NOT_SURE: 'not-sure',
  NO_HANDS: 'no-hands',
};

/* ------------------------------------------------------------------ connection */

function setConn(text, cls) {
  const pill = el('conn');
  pill.textContent = text;
  pill.className = 'pill' + (cls ? ' ' + cls : '');
}

function connect() {
  clearTimeout(retryTimer);
  setConn('connecting…', '');
  el('apiLabel').textContent = WS_URL;
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'blob';

  ws.onopen = () => {
    setConn('connected', 'ok');
    el('err').textContent = '';
    el('placeholder').style.display = 'none';
  };
  ws.onmessage = onMessage;
  ws.onerror = () => { el('err').textContent = 'websocket error — is server/ws_bridge.py running?'; };
  ws.onclose = () => {
    setConn('disconnected, retrying…', 'bad');
    el('placeholder').style.display = 'flex';
    retryTimer = setTimeout(connect, RETRY_MS);
  };
}

function send(cmd) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    el('err').textContent = 'not connected';
    return;
  }
  ws.send(JSON.stringify({ type: 'cmd', cmd }));
}

/* ------------------------------------------------------------------ messages */

function onMessage(ev) {
  if (typeof ev.data === 'string') {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'state') render(msg);
    else if (msg.type === 'error') el('err').textContent = msg.error;
    return;
  }
  showFrame(ev.data);   // Blob: JPEG preview straight from the bridge
}

function showFrame(blob) {
  const url = URL.createObjectURL(blob);
  el('preview').src = url;
  if (frameUrl) URL.revokeObjectURL(frameUrl);
  frameUrl = url;
  countFrame();
}

/* Preview rate measured in the BROWSER: the number that matters for "is it smooth?".
   The bridge's own fps field is the camera read rate, which is higher than what we get. */
let frames = 0;
let frameWindowStart = performance.now();

function countFrame() {
  frames += 1;
  const now = performance.now();
  const dt = now - frameWindowStart;
  if (dt >= 1000) {
    el('fps').textContent = Math.round((frames * 1000) / dt) + ' fps preview';
    frames = 0;
    frameWindowStart = now;
  }
}

function render(s) {
  latest = s;

  el('mode').textContent = 'mode: ' + s.mode;

  const kindEl = el('kind');
  kindEl.textContent = KIND_TEXT[s.kind] || s.kind;
  kindEl.className = 'kind ' + (KIND_CLASS[s.kind] || '');

  const labelEl = el('label');
  if (s.kind === 'WORD' || s.kind === 'LETTER') {
    labelEl.textContent = (s.label || '').toUpperCase();
    labelEl.className = 'label' + (s.stable ? '' : ' tentative');
  } else if (s.kind === 'AMBIGUOUS') {
    labelEl.textContent = 'either';
    labelEl.className = 'label tentative';
  } else {
    labelEl.textContent = '—';
    labelEl.className = 'label';
  }

  el('detail').textContent = statusLine(s);
  renderAlternatives(s.alternatives);
  renderModel('word', s.word);
  renderModel('letter', s.letter);

  const textEl = el('text');
  if (s.text) {
    textEl.textContent = s.text;
  } else {
    textEl.innerHTML = '<span class="muted">(empty)</span>';
  }
  if (s.commits && s.commits.length) flash(textEl);

  el('captureProgress').style.width = s.capture.state === 'recording'
    ? Math.round(s.capture.progress * 100) + '%'
    : '0%';
}

function statusLine(s) {
  if (s.kind === 'WORD' || s.kind === 'LETTER') {
    return (s.stable ? 'Confident (stable)' : 'Tentative — not committed yet') + ' · ' + s.reason;
  }
  if (s.kind === 'AMBIGUOUS') return 'Both models are confident — pick one with W or L, or TAB to change mode';
  if (s.kind === 'NOT_SURE') return 'Nothing above its model’s threshold · ' + s.reason;
  if (s.kind === 'NO_HANDS') return 'Show your hands, chest-up';
  return s.reason || '';
}

function renderAlternatives(alts) {
  const box = el('alternatives');
  box.innerHTML = '';
  (alts || []).forEach(([kind, label]) => {
    const b = document.createElement('button');
    b.textContent = kind + ' “' + label + '”';
    b.onclick = () => send(kind === 'WORD' ? 'word' : 'letter');
    box.appendChild(b);
  });
}

function renderModel(which, view) {
  const list = el(which + 'Top');
  const note = el(which + 'Note');
  list.innerHTML = '';
  const tops = (view && view.top) || [];
  if (!tops.length) {
    const li = document.createElement('li');
    li.className = 'stale';
    li.textContent = '—';
    list.appendChild(li);
  }
  tops.forEach(([label, score]) => {
    const li = document.createElement('li');
    li.textContent = label + '  ' + score.toFixed(3);
    if (!view.fresh) li.className = 'stale';
    list.appendChild(li);
  });
  const bits = [view && view.note ? view.note : 'waiting'];
  if (view && view.age_s != null) bits.push(view.age_s.toFixed(2) + ' s old');
  if (view && !view.fresh && view.available) bits.push('STALE — ignored by the router');
  note.textContent = bits.join(' · ');
}

let flashTimer = null;
function flash(textEl) {
  textEl.classList.add('flash');
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => textEl.classList.remove('flash'), 500);
}

/* ------------------------------------------------------------------ speech */

function speak() {
  const text = latest && latest.text;
  if (!text) {
    el('err').textContent = 'nothing committed to speak yet';
    return;
  }
  // Browser TTS: runs on the presenter's machine, never blocks the recognition loop.
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
  el('err').textContent = '';
}

/* ------------------------------------------------------------------ wiring */

el('btnMode').onclick = () => send('mode');
el('btnCapture').onclick = () => send('capture');
el('btnSpeak').onclick = speak;
el('btnConfirm').onclick = () => send('confirm');
el('btnWord').onclick = () => send('word');
el('btnLetter').onclick = () => send('letter');
el('btnSpace').onclick = () => send('space');
el('btnBackspace').onclick = () => send('backspace');
el('btnRepeat').onclick = () => send('repeat');
el('btnClear').onclick = () => send('clear');

document.addEventListener('keydown', (ev) => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;         // don't eat browser shortcuts
  const tag = (ev.target && ev.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  switch (ev.key) {
    case 'Tab': ev.preventDefault(); send('mode'); break;
    case ' ': ev.preventDefault(); send('space'); break;    // bridge: SPELL -> space, else capture
    case 'Enter': ev.preventDefault(); send('confirm'); break;
    case 'w': case 'W': send('word'); break;
    case 'l': case 'L': send('letter'); break;
    case 'r': case 'R': send('repeat'); break;
    case 'Backspace': ev.preventDefault(); send('backspace'); break;
    case 'c': case 'C': send('clear'); break;
    case 's': case 'S': ev.preventDefault(); speak(); break;
  }
});

window.addEventListener('beforeunload', () => {
  if (ws) ws.close();
});

connect();
