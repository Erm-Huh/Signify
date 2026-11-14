// Signify Rev 4 frontend: talks to the SignBridge backend (water_backend/).
// Flow: record ~3 s clip (or upload) -> POST /predict -> show top guess + "Not sure" + top-3
// -> add to sentence -> speak with the browser's speechSynthesis.

const DEFAULT_API = 'http://localhost:8000';
const RECORD_MS = 3000;

const STOPS = [
  { id: 'desk', label: 'Front desk', who: 'Receptionist',
    phrases: ["I'm here to check in", 'I have an appointment', "I'm a new patient", "Here's my insurance card", 'I feel sick'],
    replies: ['Please take a seat', 'Can I see your ID?', "You're checked in"] },
  { id: 'nurse', label: 'Nurse', who: 'Nurse',
    phrases: ['It started 2 days ago', "I'm allergic to [MEDICINE]", 'I take [MEDICATION]', 'My pain is 6 out of 10', 'I feel dizzy'],
    replies: ["Thanks, I've noted that", "I'm going to check your blood pressure", 'Any other allergies?'] },
  { id: 'doctor', label: 'Doctor', who: 'Doctor',
    phrases: ['Please slow down', 'Can you write that down?', 'What are my options?', 'I need an interpreter', 'I understand'],
    replies: ['Let me explain another way', "I'll write it down for you", 'Do you have any questions?'] },
  { id: 'out', label: 'Checkout', who: 'Checkout desk',
    phrases: ['When is my follow-up?', 'How often do I take this?', 'Can I get a printed summary?', 'Thank you'],
    replies: ['Twice a day with food', 'Your follow-up is in 2 weeks', 'Here is your printed summary'] },
];

// Friendly text for the backend's error codes (see water_backend/README.md).
const ERRORS = {
  no_hands_detected: "We couldn't see your hands. Step back so your chest and both hands are in view.",
  sign_too_short: 'That was too quick. Hold the sign a little longer and try again.',
  video_too_long: 'That clip was too long. Record just one sign (about 3 seconds).',
  file_too_large: 'That file is too big. Try a shorter clip.',
  unsupported_file_type: 'That file type is not supported. Use mp4, mov or webm.',
  unreadable_video: "We couldn't read that video. Try recording again.",
  empty_or_corrupt_video: "We couldn't read that video. Try recording again.",
  empty_file: 'The recording was empty. Try again.',
  model_unavailable: 'The sign model is not loaded on the server.',
  network: "Can't reach the sign server. Check Settings, or turn on Demo mode.",
};

const DEMO_SEQUENCE = [
  { prediction: 'hello', confidence: 0.97, confident: true, top: [['hello', 0.97], ['all', 0.01], ['open', 0.01]] },
  { prediction: 'sick', confidence: 0.81, confident: true, top: [['sick', 0.81], ['head', 0.09], ['think', 0.04]] },
  { prediction: 'head', confidence: 0.74, confident: true, top: [['head', 0.74], ['hat', 0.12], ['hair', 0.06]] },
  { prediction: 'cheek', confidence: 0.27, confident: false, top: [['cheek', 0.27], ['owie', 0.24], ['tooth', 0.18]] },
  { prediction: 'thirsty', confidence: 0.88, confident: true, top: [['thirsty', 0.88], ['drink', 0.05], ['water', 0.03]] },
  { prediction: 'thank you', confidence: 0.86, confident: true, top: [['thank you', 0.86], ['napkin', 0.06], ['bye', 0.03]] },
];

const $ = (id) => document.getElementById(id);
const state = {
  api: localStorage.getItem('signify.api') || DEFAULT_API,
  demo: localStorage.getItem('signify.demo') === '1',
  stop: 0, visited: new Set([0]),
  chips: [],            // { label, confident, top: [{label, confidence}] }
  pickIndex: null,      // chip being corrected
  threads: STOPS.map(() => []),
  stream: null, recorder: null, busy: false, demoIdx: 0,
  labels: [],
};

/* ---------- helpers ---------- */
function el(tag, props = {}, children = []) {
  const n = document.createElement(tag);
  Object.entries(props).forEach(([k, v]) => {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  });
  [].concat(children).forEach((c) => c && n.append(c));
  return n;
}
function setStatus(text, kind = '') {
  const s = $('status');
  s.className = 'status ' + kind;
  s.textContent = '';
  if (kind === 'busy') s.append(el('span', { class: 'spinner', 'aria-hidden': 'true' }));
  s.append(text);
}
const pct = (x) => Math.round(x * 100) + '%';

/* ---------- backend ---------- */
async function checkHealth() {
  const h = $('health');
  if (state.demo) { h.className = 'pill pill-wait'; h.textContent = 'Demo mode'; $('signCount').textContent = 'demo'; return; }
  try {
    const r = await fetch(state.api + '/health');
    const j = await r.json();
    if (j.model_loaded) { h.className = 'pill pill-ok'; h.textContent = 'Model ready'; loadLabels(); }
    else { h.className = 'pill pill-bad'; h.textContent = 'Model not loaded'; }
  } catch (e) {
    h.className = 'pill pill-bad'; h.textContent = 'Server offline';
  }
}

async function loadLabels() {
  try {
    const r = await fetch(state.api + '/labels');
    const j = await r.json();
    state.labels = j.labels || [];
    $('signCount').textContent = state.labels.length + ' signs';
    renderSigns();
  } catch (e) { /* list stays empty */ }
}

function demoPredict() {
  const d = DEMO_SEQUENCE[state.demoIdx++ % DEMO_SEQUENCE.length];
  return new Promise((res) => setTimeout(() => res({
    status: 'success', prediction: d.prediction, confidence: d.confidence, confident: d.confident,
    top_predictions: d.top.map(([label, confidence]) => ({ label, confidence })),
    inference_time_ms: 3200, warnings: [],
  }), 1500));
}

async function predict(blob, filename) {
  if (state.busy) return;
  state.busy = true;
  $('result').hidden = true;
  setStatus('Reading your sign… this takes a few seconds', 'busy');
  const started = performance.now();
  try {
    let j;
    if (state.demo) j = await demoPredict();
    else {
      const fd = new FormData();
      fd.append('file', blob, filename);
      let r;
      try { r = await fetch(state.api + '/predict', { method: 'POST', body: fd }); }
      catch (e) { throw { code: 'network' }; }
      j = await r.json();
      if (j.status !== 'success') throw j.error || { code: 'unknown', message: 'Something went wrong.' };
    }
    showResult(j, performance.now() - started);
  } catch (err) {
    setStatus(ERRORS[err.code] || err.message || 'Something went wrong.', 'error');
  } finally {
    state.busy = false;
  }
}

/* ---------- camera + recording ---------- */
async function toggleCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    $('preview').srcObject = null;
    $('camMsg').hidden = false; $('camBtn').textContent = 'Turn on camera'; $('recBtn').disabled = true;
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 }, audio: false });
    $('preview').srcObject = state.stream;
    $('camMsg').hidden = true; $('camBtn').textContent = 'Turn off camera'; $('recBtn').disabled = false;
    setStatus('Camera on. Press "Record a sign" when you are ready.');
  } catch (e) {
    setStatus('Camera blocked. Allow camera access, or use "Upload a clip".', 'error');
  }
}

function pickMime() {
  const options = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return options.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
}

function record() {
  if (!state.stream || state.busy) return;
  if (state.recorder) { state.recorder.stop(); return; }   // second press stops early
  const mime = pickMime();
  const rec = new MediaRecorder(state.stream, mime ? { mimeType: mime } : undefined);
  const parts = [];
  rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
  rec.onstop = () => {
    clearInterval(tick); state.recorder = null;
    $('recBar').hidden = true; $('recBtn').textContent = 'Record a sign'; $('recBtn').classList.remove('recording');
    const type = rec.mimeType || mime || 'video/webm';
    const ext = type.includes('mp4') ? 'mp4' : 'webm';
    predict(new Blob(parts, { type }), 'sign.' + ext);
  };
  state.recorder = rec;
  rec.start();
  const t0 = Date.now();
  $('recBar').hidden = false; $('recBtn').textContent = 'Stop'; $('recBtn').classList.add('recording');
  setStatus('Recording… sign now');
  const tick = setInterval(() => {
    const p = Math.min(1, (Date.now() - t0) / RECORD_MS);
    $('recFill').style.width = p * 100 + '%';
    if (p >= 1 && rec.state === 'recording') rec.stop();
  }, 100);
}

/* ---------- results + sentence ---------- */
function showResult(j, ms) {
  const top = j.top_predictions || [];
  $('result').hidden = false;
  $('resultWord').textContent = j.confident ? j.prediction : 'Not sure';
  const badge = $('resultBadge');
  badge.className = 'badge ' + (j.confident ? 'ok' : 'unsure');
  badge.textContent = j.confident ? 'Caught ✓ ' + pct(j.confidence) : 'Best guess: ' + j.prediction + ' ' + pct(j.confidence);
  $('resultMeter').style.width = pct(j.confidence);
  const warn = (j.warnings || []).length ? ' · ' + j.warnings.join(' ') : '';
  $('resultMeta').textContent = 'Took ' + (ms / 1000).toFixed(1) + ' s' + warn;
  setStatus(j.confident ? 'Added to your sentence.' : 'Not sure. Pick the right word below, or record again.');

  const chip = { label: j.prediction, confident: j.confident, top };
  state.chips.push(chip);
  const idx = state.chips.length - 1;
  renderAlts(top, idx);
  renderChips();
  const cam = $('camera'); cam.classList.add('flash'); setTimeout(() => cam.classList.remove('flash'), 350);
}

function renderAlts(top, idx) {
  const box = $('alts'); box.textContent = '';
  box.append(el('span', { class: 'hint', text: 'Did you mean:' }));
  top.forEach((t) => box.append(el('button', {
    class: 'chip', text: t.label + ' · ' + pct(t.confidence),
    onclick: () => { state.chips[idx] = { ...state.chips[idx], label: t.label, confident: true }; renderChips(); setStatus('Changed to "' + t.label + '".'); },
  })));
}

function sentence() { return state.chips.map((c) => c.label).join(' '); }

function renderChips() {
  const box = $('chips'); box.textContent = '';
  state.chips.forEach((c, i) => box.append(el('button', {
    class: 'chip' + (c.confident ? '' : ' unsure'), text: c.label,
    'aria-label': c.label + (c.confident ? '' : ', unsure. Tap to choose another guess'),
    onclick: () => { $('result').hidden = false; renderAlts(c.top, i); $('resultWord').textContent = c.label; },
  })));
  const s = $('sentenceText');
  const text = sentence();
  s.textContent = text ? text.charAt(0).toUpperCase() + text.slice(1) + '.' : 'Signs you make show up here, one at a time.';
  s.classList.toggle('empty', !text);
}

/* ---------- speech ---------- */
function loadVoices() {
  const sel = $('voiceSel'); const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en'));
  const keep = sel.value; sel.textContent = '';
  if (!voices.length) sel.append(el('option', { value: '', text: 'Default voice' }));
  voices.forEach((v) => sel.append(el('option', { value: v.name, text: v.name })));
  if (keep) sel.value = keep;
}
function speak(text) {
  if (!$('autoSpeak').checked || !text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  const v = speechSynthesis.getVoices().find((x) => x.name === $('voiceSel').value);
  if (v) u.voice = v;
  u.rate = Number($('rate').value);
  speechSynthesis.speak(u);
}

/* ---------- conversation ---------- */
function say(text) {
  if (!text) return;
  push('You', text, 'me');
  speak(text);
}
function push(who, text, kind) {
  state.threads[state.stop].push({ who, text, kind });
  renderThread(); renderLog();
}
function renderThread() {
  const box = $('thread'); box.textContent = '';
  const t = state.threads[state.stop];
  if (!t.length) box.append(el('p', { class: 'hint', text: 'What you say appears here, big enough to read across the desk.' }));
  t.forEach((m) => box.append(el('div', { class: 'msg ' + m.kind }, [el('small', { text: m.who }), m.text])));
  box.scrollTop = box.scrollHeight;
}
function renderLog() {
  const box = $('log'); box.textContent = '';
  const any = state.threads.some((t) => t.length);
  if (!any) { box.append(el('p', { class: 'hint', text: 'Everything said at each stop is kept here, so you never repeat yourself.' })); return; }
  STOPS.forEach((s, i) => {
    if (!state.threads[i].length) return;
    box.append(el('div', { class: 'log-stop' }, [el('strong', { text: s.label }),
      ...state.threads[i].map((m) => el('div', { text: m.who + ': ' + m.text }))]));
  });
}

/* ---------- stops, phrases, signs ---------- */
function renderStops() {
  const nav = $('stops'); nav.textContent = '';
  STOPS.forEach((s, i) => {
    const cur = i === state.stop;
    nav.append(el('button', {
      class: 'stop' + (cur ? ' current' : state.visited.has(i) ? ' visited' : ''),
      'aria-current': cur ? 'step' : 'false',
      onclick: () => { state.stop = i; state.visited.add(i); renderAll(); },
    }, [el('small', { text: cur ? 'Now' : state.visited.has(i) ? '✓ Visited' : 'Next' }), el('strong', { text: s.label }), el('small', { text: s.who })]));
  });
  const s = STOPS[state.stop];
  $('whoLabel').textContent = s.label; $('stopName').textContent = s.label;
  const ph = $('phrases'); ph.textContent = '';
  s.phrases.forEach((p) => ph.append(el('button', { class: 'chip phrase', text: p, onclick: () => say(p) })));
  const rp = $('replies'); rp.textContent = '';
  s.replies.forEach((p) => rp.append(el('button', { class: 'chip', text: p, onclick: () => push(s.who, p, 'them') })));
}
function renderSigns() {
  const q = $('signSearch').value.trim().toLowerCase();
  const box = $('signList'); box.textContent = '';
  state.labels.filter((l) => !q || l.label.toLowerCase().includes(q)).forEach((l) => {
    const acc = l.held_out_top1 ?? 0;
    const lvl = acc >= 0.75 ? 'hi' : acc >= 0.5 ? 'mid' : 'lo';
    box.append(el('span', { class: 'sign', title: 'Held-out accuracy ' + pct(acc) }, [el('span', { class: 'dot ' + lvl, 'aria-hidden': 'true' }), l.label]));
  });
  if (!state.labels.length) box.append(el('span', { class: 'hint', text: state.demo ? 'Connect the backend to see the full list.' : 'Loading…' }));
}
function renderAll() { renderStops(); renderThread(); renderChips(); renderLog(); }

/* ---------- wire up ---------- */
$('camBtn').onclick = toggleCamera;
$('recBtn').onclick = record;
$('fileInput').onchange = (e) => { const f = e.target.files[0]; if (f) predict(f, f.name); e.target.value = ''; };
$('sayBtn').onclick = () => { const t = $('sentenceText').classList.contains('empty') ? '' : $('sentenceText').textContent; if (t) { say(t); state.chips = []; renderChips(); $('result').hidden = true; } };
$('undoBtn').onclick = () => { state.chips.pop(); renderChips(); };
$('clearBtn').onclick = () => { state.chips = []; renderChips(); $('result').hidden = true; };
$('replyForm').onsubmit = (e) => { e.preventDefault(); const v = $('replyInput').value.trim(); if (v) { push(STOPS[state.stop].who, v, 'them'); $('replyInput').value = ''; } };
$('flipBtn').onclick = (e) => { const on = $('thread').classList.toggle('flipped'); e.target.setAttribute('aria-pressed', on); e.target.textContent = on ? 'Unflip' : 'Flip for staff'; };
$('urgentBtn').onclick = (e) => { const on = $('urgentBanner').hidden; $('urgentBanner').hidden = !on; e.target.setAttribute('aria-pressed', on); e.target.textContent = on ? 'Cancel alert' : 'I need help now'; };
$('signSearch').oninput = renderSigns;
$('copyBtn').onclick = async (e) => {
  const text = STOPS.map((s, i) => state.threads[i].length ? s.label + '\n' + state.threads[i].map((m) => m.who + ': ' + m.text).join('\n') : '').filter(Boolean).join('\n\n');
  try { await navigator.clipboard.writeText(text); e.target.textContent = 'Copied ✓'; } catch (_) { e.target.textContent = 'Copy failed'; }
  setTimeout(() => { e.target.textContent = 'Copy summary'; }, 1500);
};
$('settingsBtn').onclick = (e) => { const p = $('settings'); p.hidden = !p.hidden; e.target.setAttribute('aria-expanded', !p.hidden); };
$('apiUrl').value = state.api; $('demoMode').checked = state.demo;
$('saveSettings').onclick = () => {
  state.api = $('apiUrl').value.replace(/\/$/, '') || DEFAULT_API; state.demo = $('demoMode').checked;
  localStorage.setItem('signify.api', state.api); localStorage.setItem('signify.demo', state.demo ? '1' : '0');
  $('settings').hidden = true; checkHealth();
};
if (window.speechSynthesis) { loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; }

renderAll(); renderSigns(); checkHealth();
