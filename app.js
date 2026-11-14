const NEED = 5;
    const THRESHOLD = 0.8;
    const COOLDOWN_MS = 1500;

    let currentLabel = null;
    let count = 0;
    let sentence = [];
    let lastAccepted = 0;
    let heldLabel = null;   // the word we just accepted and are still "holding"

    function onFrame(label, confidence) {
      if (confidence >= THRESHOLD && label !== heldLabel) {
        heldLabel = null;   // gesture changed, so we are allowed to accept again
      }

      if (confidence < THRESHOLD) {
        currentLabel = null;
        count = 0;
      } else if (label === currentLabel) {
        count = count + 1;
      } else {
        currentLabel = label;
        count = 1;
      }

      if (count >= NEED) {
        const now = Date.now();
        if (now - lastAccepted > COOLDOWN_MS && heldLabel === null) {
          sentence.push(label);
          lastAccepted = now;
          heldLabel = label;
        }
        count = 0;
      }
      refresh();
    }

    function undo() {
      sentence.pop();
      refresh();
    }

    function resetAll() {
      currentLabel = null;
      count = 0;
      sentence = [];
      lastAccepted = 0;
      heldLabel = null;
      refresh();
    }

    function speak() {
      speechSynthesis.speak(new SpeechSynthesisUtterance(sentence.join(" ")));
    }

    function refresh() {
      document.getElementById("current").innerText = currentLabel || "-";
      document.getElementById("progress").innerText = count;
      document.getElementById("sentence").innerText = sentence.join(" ") || "(empty)";
    }

    let ws;

    function connect() {
      ws = new WebSocket("ws://localhost:8765");

      ws.onopen = function () {
        document.getElementById("status").innerText = "connected";
      };

      ws.onmessage = function (event) {
        const data = JSON.parse(event.data);
        document.getElementById("raw").innerText = data.label + " " + data.confidence;
        onFrame(data.label, data.confidence);
      };

      ws.onclose = function () {
        document.getElementById("status").innerText = "disconnected, retrying...";
        setTimeout(connect, 1000);
      };
    }

    connect();