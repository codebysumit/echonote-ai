<div align="center">

# 🎙️ EchoNote AI

### *Your audio. Transcribed. Understood. Remembered.*

**100% Local CPU · No API Keys · AES-256 Encrypted · Offline-First**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/Whisper-faster--whisper%20CPU-orange?style=flat-square)](https://github.com/SYSTRAN/faster-whisper)
[![Gemma](https://img.shields.io/badge/Gemma-3%201B%20GGUF%20CPU-blueviolet?style=flat-square)](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red?style=flat-square&logo=youtube)](https://youtu.be/ylCl8rsBIuw)

[**Watch Demo**](https://youtu.be/ylCl8rsBIuw) · [**Features**](#-features) · [**Setup**](#-installation) · [**Deploy**](#️-deploy-on-render)

</div>

---

## 🧠 What is EchoNote AI?

**EchoNote AI** is a professional, Claude & NotebookLM-inspired **audio intelligence studio** that converts any recording into structured, encrypted notes — all running entirely on your own laptop CPU, with zero API keys and zero internet dependency after setup.

Designed for **college students, school kids, teachers, working professionals, recipe keepers, and voice note creators**.

> *Upload a lecture. EchoNote AI transcribes it, detects what it is, writes you a tailored summary with exam questions, and lets you chat with Gemma AI about its content — forever stored privately in your encrypted local database.*

---

## 🎬 Demo

Watch EchoNote AI in action:

[![EchoNote AI Demo](https://img.youtube.com/vi/ylCl8rsBIuw/hqdefault.jpg)](https://youtu.be/ylCl8rsBIuw)

▶️ **[Watch on YouTube](https://youtu.be/ylCl8rsBIuw)**

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
- 4–8 GB RAM minimum (Gemma 3 1B Q4 fits in ~800 MB)
- ~1 GB disk space for model weights (auto-downloaded on first run)
- Optional: NVIDIA GPU (CUDA 11.x / 12.x) or Apple M-chip for faster inference

### 1. Clone the Repository
```bash
git clone https://github.com/codebysumit/echonote-ai.git
cd echonote-ai
```

### 2. Run the Smart Installer ⚡ *(recommended)*

```bash
python install.py
```

The installer **automatically detects your hardware** and picks the right `llama-cpp-python` wheel:

| Detected Hardware | llama-cpp-python Wheel | Speed |
|---|---|---|
| 🚀 NVIDIA CUDA 12.x | `/whl/cu12x` (GPU-enabled) | 5–10× faster |
| 🚀 NVIDIA CUDA 11.8 | `/whl/cu118` (GPU-enabled) | 4–8× faster |
| 🍎 Apple M1/M2/M3 | `/whl/metal` (Metal GPU) | 2–4× faster |
| 🖥️ No GPU / Unknown | `/whl/cpu` (CPU-only) | Works everywhere |

> **Why does this matter?**
> The CPU wheel (`/whl/cpu`) is **compiled without CUDA support** — even if you have an NVIDIA GPU, that wheel cannot use it.
> `install.py` detects your CUDA version and downloads the correct pre-built GPU wheel automatically.

### 3. Alternative: Standard pip install (CPU Mode)

For standard CPU systems, CI/CD pipelines, Docker, or cloud platforms (Render, Railway):
```bash
pip install -r requirements.txt
```
> *(Includes pre-compiled wheels automatically — no C++ compiler or CMake required!)*

### 4. Custom Manual GPU Install *(optional)*

**Core + Whisper:**
```bash
pip install fastapi "uvicorn[standard]" cryptography pydantic python-multipart python-dotenv
pip install faster-whisper huggingface_hub
```

**llama-cpp-python — pick ONE based on your hardware:**

```bash
# 🚀 NVIDIA CUDA 12.6
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu126

# 🚀 NVIDIA CUDA 12.1
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 🚀 NVIDIA CUDA 11.8
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# 🍎 Apple Silicon (M1/M2/M3) Metal
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

# 🖥️ CPU-only (no GPU / low-end laptop)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> 💡 Not sure which CUDA version you have? Run `nvidia-smi` in a terminal — look for `CUDA Version: XX.X` in the top-right corner.

### 5. Configure (Optional)
```bash
cp .env.example .env
```
Edit `.env`:
```env
WHISPER_MODEL_SIZE=base      # tiny / base / small
CPU_THREADS=4                # number of CPU threads for inference
MASTER_ENCRYPTION_KEY=your-secret-passphrase
```

### 6. Run EchoNote AI
```bash
python app.py
```
Open **`http://localhost:8000`** 🎉

On startup the console shows your auto-detected hardware:
```
────────────────────────────────────────────────────────────
  🚀  ECHONOTE AI — Hardware Backend
       Mode  : CUDA
       Status: NVIDIA GPU — RTX 3060 (12.0 GB VRAM) · CUDA
       GPU   : NVIDIA GeForce RTX 3060
       VRAM  : 12.0 GB
       Whisper compute : float16 on cuda
       Gemma GPU layers: -1 (all on GPU)
────────────────────────────────────────────────────────────
```

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
