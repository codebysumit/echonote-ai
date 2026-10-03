<div align="center">

# 🎙️ EchoNote AI

### *Your audio. Transcribed. Understood. Remembered.*

**100% Local CPU · No API Keys · AES-256 Encrypted · Offline-First**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/Whisper-faster--whisper%20CPU-orange?style=flat-square)](https://github.com/SYSTRAN/faster-whisper)
[![Gemma](https://img.shields.io/badge/Gemma-3%201B%20GGUF%20CPU-blueviolet?style=flat-square)](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[**Live Demo**](#-run-locally) · [**Features**](#-features) · [**Setup**](#-installation) · [**Deploy**](#️-deploy-on-render)

</div>

---

## 🧠 What is EchoNote AI?

**EchoNote AI** is a professional, Claude & NotebookLM-inspired **audio intelligence studio** that converts any recording into structured, encrypted notes — all running entirely on your own laptop CPU, with zero API keys and zero internet dependency after setup.

Designed for **college students, school kids, teachers, working professionals, recipe keepers, and voice note creators**.

> *Upload a lecture. EchoNote AI transcribes it, detects what it is, writes you a tailored summary with exam questions, and lets you chat with Gemma AI about its content — forever stored privately in your encrypted local database.*

---

## ✨ Features

| Feature | Details |
|---|---|
| ⚡ **Auto GPU / CPU Detection** | Automatically uses the best available hardware — no config needed |
| 🚀 **NVIDIA CUDA** | Whisper float16 · Gemma all layers on GPU · 5–10× faster |
| 🍎 **Apple Silicon (MPS)** | Whisper float16 on M-chip · Gemma partial GPU offload |
| 🖥️ **CPU fallback** | Whisper int8 · Gemma Q4 GGUF · works on any 4 GB RAM laptop |
| 🎧 **Multilingual Transcription** | `faster-whisper` — 99 languages + auto mixed-language detection |
| 🧠 **Local Gemma 3 1B AI** | `llama-cpp-python` GGUF Q4 — zero API keys, full offline |
| 🔒 **AES-256 Encrypted SQLite** | All transcripts, summaries & chats encrypted in `notes.db` (never in Git) |
| ✍️ **In-Browser Transcript Editor** | Live editable text with word counter, copy, and instant re-encryption on save |
| 📋 **Persistent Title Bar** | Always-visible note title, category badge, Save button and `Ctrl+S` shortcut |
| 🎙️ **Live Microphone Recording** | Record directly in the browser — no external app needed |
| ⚡ **Audio Playback + Speed Control** | Built-in player with 0.75×, 1×, 1.25×, 1.5×, 2× speed |
| 🎯 **Multi-Persona AI Summaries** | Domain-specific structured notes (see below) |
| 💬 **Gemma Grounded Chatbot** | NotebookLM-style Q&A grounded entirely in the audio transcript |
| 🧠 **Study & Exam Prep Tab** | Auto-generated high-yield exam questions with answers |
| ✅ **Action Items Checklist** | Tasks, deadlines, and decisions extracted as interactive checkboxes |
| 📥 **Export** | Download as `.md`, `.txt`, or `.json` |
| 🌗 **Dark / Light Mode** | Full glassmorphism studio UI (Claude-inspired 3-column layout) |
| 🔍 **Encrypted Full-Text Search** | Search across all note titles, summaries and transcripts |
| 🗂️ **Category Filters** | Filter notes by persona/category in the sidebar |

---

## 🎯 Multi-Persona AI Summaries

EchoNote AI automatically detects the type of audio and generates a tailored structured note:

| Persona | What You Get |
|---|---|
| 🎓 **College / University Lecture** | Core thesis, key definitions, formulas, professor tips, 5 exam Q&A pairs |
| 🏫 **School / Student Class** | Simple facts, homework tasks, vocabulary, and flashcard-ready bullets |
| 👩‍🏫 **Teacher / Educator** | Learning objectives, lesson plan breakdown, student discussion prompts |
| 💼 **Professional / Meeting** | Executive summary, decisions made, action items (who / what / when) |
| 🎙️ **Voice Note / Brainstorm** | Idea buckets, creative highlights, personal follow-up to-dos |
| 🍳 **Recipe / Culinary** | Dish name, ingredients checklist, numbered steps, chef tips |
| 🛒 **Shopping / Budget** | Itemized list, price estimates, deal highlights, priority tiers |
| 🎙️ **Interview / Podcast** | Speaker viewpoints, key quotes, topic synthesis |
| 📝 **General** | Comprehensive highlights and bullet-point overview |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- 4–8 GB RAM (for Gemma 3 1B Q4)
- ~1 GB disk space (for Whisper `base` + Gemma GGUF weights, downloaded once on first use)

### 1. Clone the Repository
```bash
git clone https://github.com/codebysumit/echonote-ai.git
cd echonote-ai
```

### 2. Install Dependencies

**FastAPI + encryption core (lightweight):**
```bash
pip install fastapi uvicorn cryptography pydantic python-multipart python-dotenv
```

**Local Whisper CPU transcription:**
```bash
pip install faster-whisper huggingface_hub
```

**Local Gemma 3 1B CPU inference (CPU-only wheel, no CUDA download):**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> 💡 Model weights (`gemma-3-1b-it-Q4_K_M.gguf` ~800 MB and Whisper `base` ~74 MB) are downloaded automatically from HuggingFace **once** on first use, then cached locally — no re-download needed.

### 3. Configure (Optional)
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to set a custom encryption passphrase or Whisper model size:
```env
WHISPER_MODEL_SIZE=base      # tiny / base / small
CPU_THREADS=2                # increase if your CPU has more cores
MASTER_ENCRYPTION_KEY=your-secret-passphrase
```

### 4. Run Locally
```bash
python app.py
```
Or with auto-reload for development:
```bash
uvicorn app:app --reload --port 8000
```

Open your browser at **`http://localhost:8000`** 🎉

> On **first run**, `notes.db` is created automatically.
> On **subsequent runs**, it connects and loads all your existing encrypted notes.

---

## 🗂️ Project Structure

```
echonote-ai/
├── app.py                          ← FastAPI server (auto-create/connect DB on startup)
├── ai_service.py                   ← Local Whisper + Gemma AI engine
├── database.py                     ← AES-256 encrypted SQLite layer
├── requirements.txt
├── .env.example
├── .gitignore                      ← notes.db, uploads/, model weights all excluded
├── Procfile                        ← Render deployment
├── render.yaml                     ← Render blueprint
├── notebooks/
│   └── smart_audio_notes_local_cpu_studio.ipynb
├── static/
│   ├── index.html                  ← 3-column Claude-style Studio UI
│   ├── css/style.css               ← Glassmorphism dark/light theme
│   └── js/app.js                   ← Full client-side studio controller
└── uploads/                        ← Audio files stored here (git-ignored)
    └── .gitkeep
```

---

## ☁️ Deploy on Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Blueprint** → select this repo.
3. Render uses the included `render.yaml` automatically.
4. Set optional environment variables in the Render dashboard:
   - `MASTER_ENCRYPTION_KEY` — your custom encryption secret
   - `WHISPER_MODEL_SIZE` — `tiny` (fastest) or `base` (recommended)

---

## 🔒 Privacy & Security

- `notes.db` is **never committed to Git** — it's in `.gitignore`.
- All note content (transcripts, summaries, chats) is **AES-256 encrypted** using Fernet + PBKDF2.
- Audio files in `uploads/` are also **git-ignored**.
- You can set a **custom master encryption key** via `.env` or the ⚙️ Settings panel.
- Everything runs **locally on your machine** — your data never leaves your device.

---

## 📄 License

[MIT License](LICENSE) — feel free to use, fork, and build on EchoNote AI.

---

<div align="center">
Made with ❤️ · Powered by <strong>faster-whisper</strong> + <strong>Gemma 3 1B</strong> + <strong>FastAPI</strong>
</div>
