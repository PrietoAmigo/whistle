"use strict";

const LANGUAGE_KEY = "whistle.language";
const KEYWORDS_KEY = "whistle.keywords";
const PAGE_SIZE = 50;

const $ = (id) => document.getElementById(id);
const recordEl = $("record");
const timerEl = $("timer");
const statusEl = $("status");
const uploadEl = $("upload");
const historyEl = $("history");
const moreEl = $("more");
const languageEl = $("language");
const keywordsEl = $("keywords");
const settingsEl = $("settings");
const settingsToggle = $("settings-toggle");
const template = $("entry-template");

// Settings are remembered in the browser. Storage can be unavailable (private browsing, blocked
// site data); the page works without it.
const store = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : JSON.parse(value);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {}
  },
};

// Finished transcripts are stored on the server, with the time their audio was recorded, the most
// recent first. Ones in flight, or failed with their audio kept for a retry, live in memory.
let saved = [];
const live = [];

const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
const newId = () => `live-${crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`}`;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function languageName(code) {
  return languageEl.querySelector(`option[value="${code}"]`)?.textContent || code;
}

function meta(entry) {
  if (entry.state) return entry.status;
  const recorded = new Date(entry.recorded_at);
  const thisYear = recorded.getFullYear() === new Date().getFullYear();
  const when = recorded.toLocaleString([], {
    year: thisYear ? undefined : "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
  const parts = [when, clock(entry.duration)];
  parts.push(entry.text ? languageName(entry.language) : "no speech");
  return parts.join(" · ");
}

function entryNode(entry) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.dataset.id = entry.id;
  node.classList.toggle("pending", entry.state === "pending");
  node.classList.toggle("failed", entry.state === "failed");
  node.querySelector(".meta").textContent = meta(entry);
  if (!entry.state) node.querySelector(".meta").title = new Date(entry.recorded_at).toLocaleString();
  node.querySelector(".text").textContent = entry.text;
  const show = (action, visible) => (node.querySelector(`[data-action="${action}"]`).hidden = !visible);
  const done = !entry.state && Boolean(entry.text);
  show("copy", done);
  show("share", done && Boolean(navigator.share));
  show("retry", entry.state === "failed");
  show("delete", entry.state !== "pending");
  return node;
}

function render() {
  historyEl.replaceChildren(...live.map(entryNode), ...saved.map(entryNode));
}

function errorMessage(error) {
  // fetch rejects with a TypeError when the network fails, or when an expired Cloudflare Access
  // session redirects the request to the login page.
  if (error instanceof TypeError) {
    return "Couldn't reach the server. Check the connection, or reload the page if you need to sign in again.";
  }
  return error.message;
}

async function transcribe(entry) {
  const form = new FormData();
  form.append("file", entry.blob, entry.name);
  if (languageEl.value) form.append("language", languageEl.value);
  if (keywordsEl.value.trim()) form.append("keywords", keywordsEl.value);
  form.append("recorded_at", entry.recordedAt);

  const response = await fetch("transcribe/stream", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().then((body) => body.detail, () => null);
    throw new Error(typeof detail === "string" ? detail : `The server answered ${response.status} ${response.statusText}`);
  }
  entry.status = "Transcribing…";
  render();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) throw new Error("The connection closed before the transcript arrived.");
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const event = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      if (!event.startsWith("data: ")) continue;
      const message = JSON.parse(event.slice(6));
      if (message.result) return message.result;
      if (message.error) throw new Error(message.error);
      if (message.partial) {
        const { text, seconds, duration } = message.partial;
        if (text) entry.text += (entry.text ? " " : "") + text;
        entry.status = `Transcribing… ${clock(seconds)} / ${clock(duration)}`;
        render();
      }
    }
  }
}

async function run(entry) {
  Object.assign(entry, { state: "pending", status: "Uploading…", text: "" });
  render();
  try {
    const result = await transcribe(entry);
    live.splice(live.indexOf(entry), 1);
    // The list is always the newest transcripts, so "Show older" can carry on from its length. An
    // upload older than everything listed waits there while there's more to load.
    const index = saved.findIndex((e) => e.recorded_at <= result.recorded_at);
    if (index >= 0) saved.splice(index, 0, result);
    else if (moreEl.hidden) saved.push(result);
    else setStatus("Saved. It's older than the transcripts shown, so it's under Show older.");
  } catch (error) {
    Object.assign(entry, { state: "failed", status: errorMessage(error) });
  }
  render();
}

function enqueue(blob, name, recordedAt) {
  const entry = { id: newId(), recordedAt: new Date(recordedAt).toISOString(), blob, name };
  live.unshift(entry);
  run(entry);
}

async function remove(entry) {
  if (live.includes(entry)) {
    live.splice(live.indexOf(entry), 1);
  } else {
    try {
      const response = await fetch(`transcripts/${entry.id}`, { method: "DELETE" });
      if (!response.ok && response.status !== 404) throw new Error(`The server answered ${response.status}`);
    } catch (error) {
      setStatus(`Couldn't delete the transcript: ${errorMessage(error)}`, true);
      return;
    }
    saved = saved.filter((e) => e !== entry);
  }
  render();
}

async function loadMore() {
  moreEl.disabled = true;
  try {
    const response = await fetch(`transcripts?limit=${PAGE_SIZE}&offset=${saved.length}`);
    if (!response.ok) throw new Error(`The server answered ${response.status}`);
    const page = await response.json();
    saved.push(...page.filter((t) => !saved.some((e) => e.id === t.id)));
    moreEl.hidden = page.length < PAGE_SIZE;
  } catch (error) {
    setStatus(`Couldn't load past transcripts: ${errorMessage(error)}`, true);
  } finally {
    moreEl.disabled = false;
  }
  render();
}

async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = Object.assign(document.createElement("textarea"), { value: text });
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  button.textContent = "Copied";
  setTimeout(() => (button.textContent = "Copy"), 1500);
}

historyEl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const id = button.closest(".entry").dataset.id;
  const entry = live.find((e) => e.id === id) || saved.find((e) => String(e.id) === id);
  if (!entry) return;
  switch (button.dataset.action) {
    case "copy":
      copy(entry.text, button);
      break;
    case "share":
      navigator.share({ text: entry.text }).catch(() => {});
      break;
    case "retry":
      run(entry);
      break;
    case "delete":
      if (confirm("Delete this transcript?")) remove(entry);
      break;
  }
});

// Recording

let session = null;

function extension(type) {
  if (type.includes("mp4")) return "m4a";
  if (type.includes("ogg")) return "ogg";
  return "webm";
}

// A ring around the button that follows the input level, so it's clear the microphone hears you.
function meter(stream) {
  try {
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(stream).connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    let level = 0;
    const draw = () => {
      if (context.state === "closed") return;
      analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((sum, v) => sum + v * v, 0) / samples.length);
      // -60 dB (a quiet room) to -10 dB (loud speech) fills the ring; it falls back gently.
      const loudness = Math.min(1, Math.max(0, (20 * Math.log10(rms + 1e-9) + 60) / 50));
      level = Math.max(loudness, level * 0.85);
      recordEl.style.setProperty("--level", level.toFixed(3));
      requestAnimationFrame(draw);
    };
    draw();
    return context;
  } catch {
    return null;
  }
}

async function start() {
  let stream;
  recordEl.disabled = true; // while the browser asks for the microphone
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    setStatus(error.name === "NotAllowedError"
      ? "Microphone access is blocked. Allow it in the browser's site settings."
      : `Can't use the microphone: ${error.message}`, true);
    return;
  } finally {
    recordEl.disabled = false;
  }
  const mimeType = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"].find((type) => MediaRecorder.isTypeSupported(type));
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) chunks.push(event.data);
  });
  recorder.addEventListener("stop", () => {
    const type = recorder.mimeType || mimeType || "audio/webm";
    enqueue(new Blob(chunks, { type }), `recording.${extension(type)}`, started);
  });
  recorder.start(1000);
  const started = Date.now();

  const tick = () => (timerEl.textContent = clock((Date.now() - started) / 1000));
  session = { recorder, stream, timer: setInterval(tick, 250), meter: meter(stream), wakeLock: null };
  tick();
  // Keep the screen on, since a locked phone can stop the recording.
  navigator.wakeLock?.request("screen").then((lock) => (session ? (session.wakeLock = lock) : lock.release()), () => {});
  recordEl.classList.add("recording");
  recordEl.setAttribute("aria-label", "Stop recording");
  setStatus("Recording… tap to stop");
}

function stop() {
  const { recorder, stream, timer, meter, wakeLock } = session;
  session = null;
  clearInterval(timer);
  meter?.close().catch(() => {});
  wakeLock?.release().catch(() => {});
  recorder.stop();
  stream.getTracks().forEach((track) => track.stop());
  recordEl.classList.remove("recording");
  recordEl.setAttribute("aria-label", "Start recording");
  setStatus("Tap to record");
}

recordEl.addEventListener("click", () => (session ? stop() : start()));
moreEl.addEventListener("click", loadMore);

uploadEl.addEventListener("change", () => {
  const [file] = uploadEl.files;
  // A file's last change is the best guess at when it was recorded.
  if (file) enqueue(file, file.name, file.lastModified || Date.now());
  uploadEl.value = "";
});

// Settings

settingsToggle.addEventListener("click", () => {
  const open = settingsEl.hidden;
  settingsEl.hidden = !open;
  settingsToggle.setAttribute("aria-expanded", String(open));
});
languageEl.value = store.get(LANGUAGE_KEY, "");
keywordsEl.value = store.get(KEYWORDS_KEY, "");
languageEl.addEventListener("change", () => store.set(LANGUAGE_KEY, languageEl.value));
keywordsEl.addEventListener("input", () => store.set(KEYWORDS_KEY, keywordsEl.value));

// Browsers only allow the microphone on HTTPS (or localhost).
if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
  recordEl.disabled = true;
  setStatus(window.isSecureContext
    ? "This browser can't record audio here. Upload a recording instead."
    : "Browsers only allow the microphone over HTTPS, so open this page's https:// address to record. Uploading a recording works here too.");
}

loadMore();
