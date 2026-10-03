/**
 * Smart Audio Notes - Studio Client Application (100% Local CPU Mode)
 * Audio Intelligence, Live In-Browser Transcript Editor,
 * Multi-Persona Summarizer, and Local Gemma AI Copilot.
 */

// Application State
const state = {
  activeNoteId: null,
  activeNote: null,
  activeTab: "summary",
  notes: [],
  categories: {},
  selectedCategory: "all",
  searchQuery: "",
  
  // Local Master Encryption Key
  encryptionKey: localStorage.getItem("SAN_ENC_KEY") || "",
  
  // Audio Recording State
  mediaRecorder: null,
  audioChunks: [],
  recordingInterval: null,
  recordingSeconds: 0,
  recordedBlob: null
};

// DOM Ready initialization
document.addEventListener("DOMContentLoaded", async () => {
  setupTheme();
  fetchHardwareBadge();
  await loadCategories();
  await loadNotesList();
  setupEventListeners();
  loadSavedSettingsIntoForm();
});

// ─── Hardware Badge ───────────────────────────────────────────────────────────

/**
 * Fetches /api/hardware and updates the header badge with the real backend.
 * Color: 🟢 CUDA (green) | 🍎 MPS (purple) | 🖥️ CPU (grey)
 */
async function fetchHardwareBadge() {
  const badge = document.getElementById("hw-badge");
  if (!badge) return;
  try {
    const res  = await fetch("/api/hardware");
    const hw   = await res.json();
    const icon = { cuda: "🚀", mps: "🍎", cpu: "🖥️" }[hw.backend] || "🖥️";
    const color = { cuda: "#3fb950", mps: "#a78bfa", cpu: "var(--text-muted)" }[hw.backend] || "var(--text-muted)";
    const label = hw.backend === "cuda"
      ? `${icon} GPU: ${hw.gpu_name} (${hw.vram_gb} GB) · CUDA`
      : hw.backend === "mps"
        ? `${icon} Apple Silicon GPU · float16`
        : `${icon} CPU · int8 · ${hw.cpu_threads} threads`;
    badge.textContent  = label;
    badge.style.color  = color;
    badge.style.borderColor = color;
    badge.title = hw.summary;
  } catch (_) {
    if (badge) badge.textContent = "🖥️ Local Mode";
  }
}

function setupTheme() {
  const savedTheme = localStorage.getItem("SAN_THEME") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(savedTheme);
}

// ─── Title Bar Sync ───────────────────────────────────────────────────────────

/**
 * Called when the user types in the global title bar input.
 * Syncs the workspace note-title input and marks unsaved changes.
 */
function syncTitleFromTitlebar(value) {
  const workspaceInput = document.getElementById("note-title-input");
  if (workspaceInput) workspaceInput.value = value;
  markTitlebarUnsaved(true);
}

/**
 * Updates the global title bar to reflect the currently active note.
 */
function updateTitleBar(note) {
  const titleInput   = document.getElementById("titlebar-note-title");
  const saveBtn      = document.getElementById("titlebar-save-btn");
  const catPill      = document.getElementById("titlebar-category-pill");
  const langPill     = document.getElementById("titlebar-lang-pill");

  if (!note) {
    if (titleInput) { titleInput.value = ""; titleInput.placeholder = "Select or create a note to begin..."; }
    if (saveBtn)    saveBtn.style.display = "none";
    if (catPill)    catPill.style.display = "none";
    if (langPill)   langPill.style.display = "none";
    markTitlebarUnsaved(false);
    return;
  }

  if (titleInput) titleInput.value = note.title || "";

  const catInfo = state.categories[note.category] || { name: note.category };
  if (catPill) {
    catPill.textContent = catInfo.name;
    catPill.style.display = "inline-block";
  }

  if (langPill && note.language && note.language !== "auto") {
    langPill.textContent = "🌐 " + note.language.toUpperCase();
    langPill.style.display = "inline-block";
  } else if (langPill) {
    langPill.style.display = "none";
  }

  if (saveBtn) saveBtn.style.display = "inline-flex";
  markTitlebarUnsaved(false);
}

/** Shows/hides the unsaved-changes dot in the title bar. */
function markTitlebarUnsaved(unsaved) {
  const dot = document.getElementById("titlebar-unsaved-dot");
  if (dot) dot.style.display = unsaved ? "inline-block" : "none";
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("SAN_THEME", next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  const btn = document.getElementById("theme-toggle-btn");
  if (btn) {
    btn.innerHTML = theme === "dark" ? "☀️" : "🌙";
  }
}

// API Utilities with Custom Encryption Headers
function getHeaders(extra = {}) {
  const headers = { ...extra };
  if (state.encryptionKey) {
    headers["X-Encryption-Key"] = state.encryptionKey;
  }
  return headers;
}

// Categories & Notes Loading
async function loadCategories() {
  try {
    const res = await fetch("/api/categories");
    if (res.ok) {
      state.categories = await res.json();
      renderCategoryFilters();
      populateCategorySelects();
    }
  } catch (err) {
    console.warn("Could not load categories:", err);
  }
}

function renderCategoryFilters() {
  const container = document.getElementById("category-filter-bar");
  if (!container) return;

  let html = `<button class="cat-pill active" onclick="filterCategory('all')">✨ All Notes</button>`;
  for (const [key, val] of Object.entries(state.categories)) {
    html += `<button class="cat-pill" data-cat="${key}" onclick="filterCategory('${key}')">${val.name.split(' ')[0]} ${val.name.split(' ')[1] || ''}</button>`;
  }
  container.innerHTML = html;
}

function populateCategorySelects() {
  const selectUpload = document.getElementById("upload-category-select");
  const selectRegen = document.getElementById("regen-category-select");

  let optionsHtml = `<option value="auto">🤖 Auto-Detect with Local Gemma</option>`;
  for (const [key, val] of Object.entries(state.categories)) {
    optionsHtml += `<option value="${key}">${val.name}</option>`;
  }

  if (selectUpload) selectUpload.innerHTML = optionsHtml;
  if (selectRegen) selectRegen.innerHTML = optionsHtml;
}

function filterCategory(cat) {
  state.selectedCategory = cat;
  document.querySelectorAll(".cat-pill").forEach(pill => {
    if (pill.getAttribute("data-cat") === cat || (cat === "all" && !pill.getAttribute("data-cat"))) {
      pill.classList.add("active");
    } else {
      pill.classList.remove("active");
    }
  });
  loadNotesList();
}

async function loadNotesList() {
  try {
    const url = new URL("/api/notes", window.location.origin);
    if (state.searchQuery) url.searchParams.set("q", state.searchQuery);
    if (state.selectedCategory && state.selectedCategory !== "all") {
      url.searchParams.set("category", state.selectedCategory);
    }

    const res = await fetch(url, { headers: getHeaders() });
    if (res.ok) {
      const data = await res.json();
      state.notes = data.notes || [];
      renderNotesSidebar();
    }
  } catch (err) {
    showToast("Failed to load notes", "error");
  }
}

function renderNotesSidebar() {
  const listEl = document.getElementById("notes-list");
  if (!listEl) return;

  if (state.notes.length === 0) {
    listEl.innerHTML = `
      <div style="text-align: center; padding: 30px 10px; color: var(--text-muted); font-size: 0.85rem;">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🎙️</div>
        No audio notes found.<br>Click <strong>+ New Audio Note</strong> to start!
      </div>`;
    return;
  }

  listEl.innerHTML = state.notes.map(note => {
    const isActive = note.id === state.activeNoteId ? "active" : "";
    const catInfo = state.categories[note.category] || { name: note.category };
    const dateStr = new Date(note.updated_at || note.created_at).toLocaleDateString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });

    return `
      <div class="note-item-card ${isActive}" onclick="selectNote('${note.id}')">
        <div class="note-item-title">${escapeHtml(note.title || "Untitled Note")}</div>
        <div class="note-item-preview">${escapeHtml(note.summary_preview || "Encrypted Audio Note")}</div>
        <div class="note-item-meta">
          <span class="note-badge-tag">${catInfo.name.split(' ')[0]} ${note.category}</span>
          <span>${dateStr}</span>
        </div>
      </div>
    `;
  }).join("");
}

// Select and Display Active Note
async function selectNote(noteId) {
  state.activeNoteId = noteId;
  renderNotesSidebar();

  try {
    const res = await fetch(`/api/notes/${noteId}`, { headers: getHeaders() });
    if (res.ok) {
      state.activeNote = await res.json();
      renderWorkspace();
    } else {
      showToast("Could not load note content", "error");
    }
  } catch (err) {
    showToast("Error loading note", "error");
  }
}

function renderWorkspace() {
  const emptyState = document.getElementById("empty-workspace-state");
  const activeWorkspace = document.getElementById("active-note-workspace");
  if (!emptyState || !activeWorkspace) return;

  if (!state.activeNote) {
    emptyState.style.display = "flex";
    activeWorkspace.style.display = "none";
    updateTitleBar(null);
    return;
  }

  emptyState.style.display = "none";
  activeWorkspace.style.display = "flex";

  // Sync global title bar
  updateTitleBar(state.activeNote);

  const titleInput = document.getElementById("note-title-input");
  if (titleInput) titleInput.value = state.activeNote.title || "";

  const catBadge = document.getElementById("note-category-badge");
  const catInfo = state.categories[state.activeNote.category] || { name: state.activeNote.category };
  if (catBadge) catBadge.textContent = catInfo.name;

  const audioContainer = document.getElementById("audio-player-wrapper");
  const audioElem = document.getElementById("main-audio-element");
  if (state.activeNote.audio_filename) {
    audioContainer.style.display = "flex";
    audioElem.src = `/audio/${state.activeNote.audio_filename}`;
  } else {
    audioContainer.style.display = "none";
  }

  const transcriptEditor = document.getElementById("transcript-editor");
  if (transcriptEditor) {
    transcriptEditor.value = state.activeNote.transcript || "";
    updateWordCount();
  }

  renderStructuredSummary();
  renderChatMessages();
}

function switchWorkspaceTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab-btn").forEach(btn => {
    if (btn.getAttribute("data-tab") === tab) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  const summaryView = document.getElementById("tab-view-summary");
  const editorView = document.getElementById("tab-view-editor");
  const studyView = document.getElementById("tab-view-study");
  const actionsView = document.getElementById("tab-view-actions");

  if (summaryView) summaryView.style.display = tab === "summary" ? "block" : "none";
  if (editorView) editorView.style.display = tab === "editor" ? "block" : "none";
  if (studyView) studyView.style.display = tab === "study" ? "block" : "none";
  if (actionsView) actionsView.style.display = tab === "actions" ? "block" : "none";
}

function renderStructuredSummary() {
  const summaryBox = document.getElementById("summary-text-box");
  const studyBox = document.getElementById("study-guide-box");
  const actionsBox = document.getElementById("actions-list-box");

  if (!state.activeNote) return;

  const struct = state.activeNote.structured_data || {};
  
  if (summaryBox) {
    summaryBox.innerHTML = renderMarkdown(state.activeNote.summary || "No summary available.");
  }

  if (studyBox) {
    const questions = struct.study_exam_questions || [];
    if (questions.length > 0) {
      let qHtml = `<div style="display: flex; flex-direction: column; gap: 14px;">`;
      questions.forEach((q, i) => {
        qHtml += `
          <div style="background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 16px;">
            <div style="font-weight: 700; color: var(--accent-primary); margin-bottom: 6px;">🧠 Question ${i+1}: ${escapeHtml(q.question)}</div>
            <div style="font-size: 0.9rem; color: var(--text-primary); line-height: 1.5;"><strong>Answer:</strong> ${escapeHtml(q.answer)}</div>
          </div>
        `;
      });
      qHtml += `</div>`;
      studyBox.innerHTML = qHtml;
    } else {
      studyBox.innerHTML = `
        <div style="text-align: center; padding: 30px; color: var(--text-muted);">
          No exam flashcards detected.<br>
          <button class="chip-btn" style="margin-top: 10px;" onclick="sendQuickPrompt('Generate 5 high-yield exam questions with answers based on this lecture')">Generate 5 Exam Questions</button>
        </div>`;
    }
  }

  if (actionsBox) {
    const actions = struct.action_items || [];
    if (actions.length > 0) {
      let actHtml = `<div style="display: flex; flex-direction: column; gap: 10px;">`;
      actions.forEach((act, i) => {
        actHtml += `
          <label style="display: flex; align-items: flex-start; gap: 10px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 12px 14px; cursor: pointer;">
            <input type="checkbox" style="margin-top: 4px; accent-color: var(--accent-primary);" />
            <span style="font-size: 0.9rem; color: var(--text-primary); line-height: 1.4;">${escapeHtml(act)}</span>
          </label>
        `;
      });
      actHtml += `</div>`;
      actionsBox.innerHTML = actHtml;
    } else {
      actionsBox.innerHTML = `
        <div style="text-align: center; padding: 30px; color: var(--text-muted);">
          No explicit action items found.<br>
          <button class="chip-btn" style="margin-top: 10px;" onclick="sendQuickPrompt('Extract all tasks, homework assignments, and deadlines mentioned in the audio')">Extract Tasks & Deadlines</button>
        </div>`;
    }
  }
}

// Inbuilt Live Transcript Editor
function updateWordCount() {
  const text = document.getElementById("transcript-editor")?.value || "";
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const chars = text.length;
  const counter = document.getElementById("editor-word-count");
  if (counter) counter.textContent = `${words} words · ${chars} chars`;
}

async function saveTranscriptChanges() {
  if (!state.activeNoteId) return;
  const newTranscript = document.getElementById("transcript-editor")?.value || "";
  // Prefer the title bar input (always visible) over the workspace input
  const titlebarInput = document.getElementById("titlebar-note-title");
  const workspaceInput = document.getElementById("note-title-input");
  const newTitle = (titlebarInput?.value || workspaceInput?.value || state.activeNote.title).trim();

  try {
    const res = await fetch(`/api/notes/${state.activeNoteId}`, {
      method: "PUT",
      headers: getHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        title: newTitle,
        transcript: newTranscript
      })
    });

    if (res.ok) {
      const data = await res.json();
      state.activeNote = data.note;
      updateTitleBar(state.activeNote);
      showToast("Changes saved & re-encrypted in SQLite", "success");
      loadNotesList();
    } else {
      showToast("Save failed", "error");
    }
  } catch (err) {
    showToast("Error saving changes", "error");
  }
}

// Regenerate AI Summary
function openRegenerateModal() {
  const modal = document.getElementById("regenerate-modal");
  if (modal) modal.classList.add("active");
}

async function handleRegenerateSummary() {
  if (!state.activeNoteId) return;
  const cat = document.getElementById("regen-category-select")?.value || "auto";
  const customInst = document.getElementById("regen-instructions-input")?.value || "";

  closeAllModals();
  showToast("Re-analyzing with local CPU Gemma...", "info");

  try {
    const res = await fetch(`/api/notes/${state.activeNoteId}/regenerate-summary`, {
      method: "POST",
      headers: getHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        category: cat,
        custom_instructions: customInst
      })
    });

    if (res.ok) {
      const data = await res.json();
      state.activeNote = data.note;
      renderWorkspace();
      showToast("Summary updated with new persona!", "success");
      loadNotesList();
    } else {
      showToast("Failed to regenerate summary", "error");
    }
  } catch (err) {
    showToast("Error regenerating summary", "error");
  }
}

// Grounded Local Chatbot
function renderChatMessages() {
  const container = document.getElementById("chat-messages");
  if (!container || !state.activeNote) return;

  const chats = state.activeNote.chats || [];
  if (chats.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; color: var(--text-muted); padding: 40px 10px; font-size: 0.85rem;">
        <div style="font-size: 2rem; margin-bottom: 8px;">💡</div>
        <strong>Gemma Audio Assistant (Local CPU)</strong><br>
        Ask any question grounded in this audio note. Key points, quiz questions, action items, or clarifications.
      </div>
    `;
    return;
  }

  container.innerHTML = chats.map(c => `
    <div class="chat-msg ${c.role}">
      <div class="chat-bubble">${renderMarkdown(c.content)}</div>
    </div>
  `).join("");

  container.scrollTop = container.scrollHeight;
}

async function sendChatMessage() {
  if (!state.activeNoteId) {
    showToast("Please select an audio note first", "info");
    return;
  }

  const input = document.getElementById("chat-input");
  const question = input?.value.trim();
  if (!question) return;

  input.value = "";

  if (!state.activeNote.chats) state.activeNote.chats = [];
  state.activeNote.chats.push({ role: "user", content: question });
  renderChatMessages();

  const container = document.getElementById("chat-messages");
  const thinkingId = "thinking-indicator-" + Date.now();
  if (container) {
    container.insertAdjacentHTML("beforeend", `
      <div class="chat-msg assistant" id="${thinkingId}">
        <div class="chat-bubble" style="color: var(--text-muted);">✨ Gemma is thinking on CPU...</div>
      </div>
    `);
    container.scrollTop = container.scrollHeight;
  }

  try {
    const res = await fetch(`/api/notes/${state.activeNoteId}/chat`, {
      method: "POST",
      headers: getHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question: question })
    });

    const thinkingElem = document.getElementById(thinkingId);
    if (thinkingElem) thinkingElem.remove();

    if (res.ok) {
      const data = await res.json();
      state.activeNote.chats.push({
        role: "assistant",
        content: data.assistant_message.content
      });
      renderChatMessages();
    } else {
      showToast("Failed to get answer", "error");
    }
  } catch (err) {
    const thinkingElem = document.getElementById(thinkingId);
    if (thinkingElem) thinkingElem.remove();
    showToast("Chat request failed", "error");
  }
}

function sendQuickPrompt(promptText) {
  const input = document.getElementById("chat-input");
  if (input) {
    input.value = promptText;
    sendChatMessage();
  }
}

// Audio Upload & Processing
function openUploadModal() {
  const modal = document.getElementById("upload-modal");
  if (modal) modal.classList.add("active");
}

async function handleAudioUpload() {
  const fileInput = document.getElementById("audio-file-input");
  const file = fileInput?.files[0] || state.recordedBlob;

  if (!file) {
    showToast("Please choose an audio file or record audio first", "warning");
    return;
  }

  const category = document.getElementById("upload-category-select")?.value || "auto";
  const language = document.getElementById("upload-language-select")?.value || "auto";

  const formData = new FormData();
  formData.append("file", file, file.name || "recording.webm");
  formData.append("category", category);
  formData.append("language", language);

  closeAllModals();
  showToast("Transcribing on local CPU & encrypting...", "info");

  try {
    const res = await fetch("/api/upload-audio", {
      method: "POST",
      headers: getHeaders(),
      body: formData
    });

    if (res.ok) {
      const data = await res.json();
      showToast("Audio processed & saved locally!", "success");
      await loadNotesList();
      if (data.note) {
        selectNote(data.note.id);
      }
      state.recordedBlob = null;
    } else {
      showToast("Audio upload failed", "error");
    }
  } catch (err) {
    showToast("Network error uploading audio", "error");
  }
}

// Live Microphone Recording
async function toggleLiveRecording() {
  const recordBtn = document.getElementById("mic-record-btn");
  const timerElem = document.getElementById("record-timer");

  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    state.mediaRecorder.stop();
    clearInterval(state.recordingInterval);
    recordBtn.classList.remove("recording");
    recordBtn.innerHTML = "🎙️";
    showToast("Audio recording completed. Ready to process!", "success");
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.audioChunks = [];
      state.mediaRecorder = new MediaRecorder(stream);

      state.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };

      state.mediaRecorder.onstop = () => {
        state.recordedBlob = new Blob(state.audioChunks, { type: "audio/webm" });
        stream.getTracks().forEach(track => track.stop());
        const previewAudio = document.getElementById("record-preview-audio");
        if (previewAudio) {
          previewAudio.src = URL.createObjectURL(state.recordedBlob);
          previewAudio.style.display = "block";
        }
      };

      state.mediaRecorder.start();
      state.recordingSeconds = 0;
      recordBtn.classList.add("recording");
      recordBtn.innerHTML = "⏹️";

      state.recordingInterval = setInterval(() => {
        state.recordingSeconds++;
        const mins = String(Math.floor(state.recordingSeconds / 60)).padStart(2, "0");
        const secs = String(state.recordingSeconds % 60).padStart(2, "0");
        if (timerElem) timerElem.textContent = `${mins}:${secs}`;
      }, 1000);

      showToast("Recording live audio...", "info");
    } catch (err) {
      showToast("Microphone access denied or unsupported", "error");
    }
  }
}

// Audio Speed Controls
function changeAudioSpeed() {
  const audio = document.getElementById("main-audio-element");
  const btn = document.getElementById("audio-speed-btn");
  if (!audio || !btn) return;

  const speeds = [1, 1.25, 1.5, 2, 0.75];
  const currentSpeed = audio.playbackRate || 1;
  const nextIdx = (speeds.indexOf(currentSpeed) + 1) % speeds.length;
  const newSpeed = speeds[nextIdx];

  audio.playbackRate = newSpeed;
  btn.textContent = `${newSpeed}x`;
}

// Export Note Options
function openExportModal() {
  if (!state.activeNoteId) {
    showToast("Please select a note to export", "warning");
    return;
  }
  const modal = document.getElementById("export-modal");
  if (modal) modal.classList.add("active");
}

function triggerExport(format) {
  if (!state.activeNoteId) return;
  const url = `/api/notes/${state.activeNoteId}/export/${format}`;
  window.open(url, "_blank");
  closeAllModals();
  showToast(`Exported as .${format}`, "success");
}

// Local Settings Modal
function openSettingsModal() {
  const modal = document.getElementById("settings-modal");
  if (modal) modal.classList.add("active");
}

function loadSavedSettingsIntoForm() {
  const encInput = document.getElementById("settings-enc-key");
  if (encInput) encInput.value = state.encryptionKey;
}

function saveSettings() {
  const encVal = document.getElementById("settings-enc-key")?.value.trim() || "";
  state.encryptionKey = encVal;
  localStorage.setItem("SAN_ENC_KEY", encVal);

  closeAllModals();
  showToast("Local settings saved!", "success");
  loadNotesList();
}

// Delete Active Note
async function deleteActiveNote() {
  if (!state.activeNoteId) return;
  if (!confirm("Are you sure you want to permanently delete this audio note?")) return;

  try {
    const res = await fetch(`/api/notes/${state.activeNoteId}`, {
      method: "DELETE",
      headers: getHeaders()
    });

    if (res.ok) {
      showToast("Note deleted", "info");
      state.activeNoteId = null;
      state.activeNote = null;
      renderWorkspace();
      loadNotesList();
    } else {
      showToast("Delete failed", "error");
    }
  } catch (err) {
    showToast("Error deleting note", "error");
  }
}

// Helper: Setup Global Event Listeners
function setupEventListeners() {
  const searchInput = document.getElementById("notes-search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      state.searchQuery = e.target.value;
      loadNotesList();
    });
  }

  const chatInput = document.getElementById("chat-input");
  if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  const dropzone = document.getElementById("audio-dropzone");
  const fileInput = document.getElementById("audio-file-input");
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        document.getElementById("dropzone-label").textContent = `Selected: ${e.dataTransfer.files[0].name}`;
      }
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) {
        document.getElementById("dropzone-label").textContent = `Selected: ${fileInput.files[0].name}`;
      }
    });
  }

  const transcriptEditor = document.getElementById("transcript-editor");
  if (transcriptEditor) {
    transcriptEditor.addEventListener("input", updateWordCount);
  }

  // Mark unsaved whenever the workspace title is edited
  const workspaceTitleInput = document.getElementById("note-title-input");
  if (workspaceTitleInput) {
    workspaceTitleInput.addEventListener("input", (e) => {
      const titlebarInput = document.getElementById("titlebar-note-title");
      if (titlebarInput) titlebarInput.value = e.target.value;
      markTitlebarUnsaved(true);
    });
  }

  // Global Ctrl+S shortcut → save active note
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      if (state.activeNoteId) saveTranscriptChanges();
    }
  });
}

function closeAllModals() {
  document.querySelectorAll(".modal-overlay").forEach(m => m.classList.remove("active"));
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function renderMarkdown(md) {
  if (!md) return "";
  let html = escapeHtml(md);

  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
  html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');
  html = html.replace(/- \[ \] (.*$)/gim, '<div style="margin: 4px 0;"><input type="checkbox" disabled /> $1</div>');
  html = html.replace(/- \[x\] (.*$)/gim, '<div style="margin: 4px 0;"><input type="checkbox" checked disabled /> $1</div>');
  html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/\n\n/gim, '<br/><br/>');

  return html;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
