"""
EchoNote AI — Local Inference Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Auto-Detection Hardware Priority:
  1. NVIDIA GPU  → faster-whisper runs float16 CUDA | llama-cpp uses n_gpu_layers=-1
  2. Apple MPS   → faster-whisper float16 on MPS   | llama-cpp n_gpu_layers=1
  3. CPU fallback→ faster-whisper int8 CPU          | llama-cpp CPU only (default)

Zero API keys. 100% offline after first model download.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

# ─── PyAV Compatibility Patch ─────────────────────────────────────────────────
# Recent PyAV releases removed the 'metadata_errors' parameter from av.open().
# faster-whisper calls av.open(..., metadata_errors="ignore"), causing a TypeError.
# This patch intercepts av.open and removes 'metadata_errors' safely.
try:
    import av
    if not hasattr(av, "_original_open"):
        av._original_open = av.open

    def patched_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return av._original_open(*args, **kwargs)

    av.open = patched_open
except Exception:
    pass

# ─── Model Singletons (lazy-loaded) ───────────────────────────────────────────
_WHISPER_MODEL = None
_LLAMA_MODEL   = None

# ─── Config from .env (with sensible defaults) ────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
CPU_THREADS        = int(os.getenv("CPU_THREADS", "2"))
GGUF_REPO_ID       = os.getenv("GGUF_REPO_ID",   "unsloth/gemma-3-1b-it-GGUF")
GGUF_FILENAME      = os.getenv("GGUF_FILENAME",   "gemma-3-1b-it-Q4_K_M.gguf")


# ══════════════════════════════════════════════════════════════════════════════
# ①  Hardware Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_hardware() -> Dict[str, Any]:
    """
    Probes available hardware and returns the best backend for Whisper and llama-cpp.

    Returns a dict:
        {
          "backend":          "cuda" | "mps" | "cpu",
          "whisper_device":   "cuda" | "auto" | "cpu",
          "whisper_compute":  "float16" | "int8",
          "llama_n_gpu_layers": int,   # -1 = all layers on GPU, 0 = CPU only
          "gpu_name":         str | None,
          "vram_gb":          float | None,
          "summary":          str          # human-readable one-liner
        }
    """
    hw = {
        "backend":            "cpu",
        "whisper_device":     "cpu",
        "whisper_compute":    "int8",
        "llama_n_gpu_layers": 0,
        "gpu_name":           None,
        "vram_gb":            None,
        "summary":            "CPU-only mode (no GPU detected)"
    }

    # ── Try NVIDIA CUDA ──────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name  = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb    = round(vram_bytes / (1024 ** 3), 1)
            hw.update({
                "backend":            "cuda",
                "whisper_device":     "cuda",
                "whisper_compute":    "float16",
                "llama_n_gpu_layers": -1,          # all layers on GPU
                "gpu_name":           gpu_name,
                "vram_gb":            vram_gb,
                "summary":            f"NVIDIA GPU — {gpu_name} ({vram_gb} GB VRAM) · CUDA"
            })
            return hw
    except ImportError:
        pass  # torch not installed → skip CUDA check

    # ── Try Apple MPS (M1/M2/M3) ────────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            hw.update({
                "backend":            "mps",
                "whisper_device":     "auto",      # faster-whisper auto picks MPS
                "whisper_compute":    "float16",
                "llama_n_gpu_layers": 1,           # partial offload (MPS limitation)
                "gpu_name":           "Apple Silicon (MPS)",
                "vram_gb":            None,
                "summary":            "Apple Silicon GPU (MPS) · float16"
            })
            return hw
    except (ImportError, AttributeError):
        pass

    # ── CPU fallback ────────────────────────────────────────────────────────
    cpu_cores = os.cpu_count() or 1
    hw["summary"] = f"CPU-only mode · {cpu_cores} logical cores · int8 quantized"
    return hw


# Run detection once at module import — cheap (< 50 ms)
HW = detect_hardware()

def _print_hardware_banner():
    icon = {"cuda": "🚀", "mps": "🍎", "cpu": "🖥️"}.get(HW["backend"], "🖥️")
    print("─" * 60)
    print(f"  {icon}  ECHONOTE AI — Hardware Backend")
    print(f"       Mode  : {HW['backend'].upper()}")
    print(f"       Status: {HW['summary']}")
    if HW["gpu_name"]:
        print(f"       GPU   : {HW['gpu_name']}")
    if HW["vram_gb"]:
        print(f"       VRAM  : {HW['vram_gb']} GB")
    print(f"       Whisper compute : {HW['whisper_compute']} on {HW['whisper_device']}")
    print(f"       Gemma GPU layers: {HW['llama_n_gpu_layers']} "
          f"({'all on GPU' if HW['llama_n_gpu_layers'] == -1 else 'CPU only' if HW['llama_n_gpu_layers'] == 0 else 'partial GPU'})")
    print("─" * 60)

_print_hardware_banner()


# ══════════════════════════════════════════════════════════════════════════════
# ②  Persona Definitions
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_PERSONAS = {
    "college": {
        "name": "🎓 College / University Lecture",
        "description": "Deep academic breakdown with core concepts, theories, professor tips, glossary, and exam questions.",
        "focus": "Academic depth, core thesis, key definitions, formulas, homework/deadlines, and 5 potential exam questions."
    },
    "school": {
        "name": "🏫 School / Student Class",
        "description": "Clear, easy-to-understand student notes, simple explanations, flashcards, and homework tasks.",
        "focus": "Simple clarity, key facts, homework assignments, vocabulary, and quick revision flashcards."
    },
    "teacher": {
        "name": "👩‍🏫 Teacher / Educator Lesson",
        "description": "Pedagogical summary, learning objectives, student discussion points, and assignment planning.",
        "focus": "Curriculum objectives, core lesson takeaways, student evaluation questions, and classroom activities."
    },
    "professional": {
        "name": "💼 Professional / Meeting & Standup",
        "description": "Executive summary, strategic decisions, action items with owners and deadlines, and next steps.",
        "focus": "Executive summary, meeting purpose, agreed decisions, prioritized action items (Who/What/When), next milestones."
    },
    "voicenote": {
        "name": "🎙️ Personal Voice Note & Brainstorm",
        "description": "Synthesized thoughts, creative ideas, categorized brainstorm, and personal to-do list.",
        "focus": "Core insight, structured thought buckets, creative takeaways, and personal follow-ups."
    },
    "interview": {
        "name": "🎙️ Interview / Podcast / Conversation",
        "description": "Q&A synthesis, speaker viewpoints, memorable quotes, and subject assessment.",
        "focus": "Main interview themes, question-by-question synthesis, notable quotes, and overall impressions."
    },
    "recipe": {
        "name": "🍳 Cooking Recipe & Culinary",
        "description": "Prep time, ingredients checklist with quantities, step-by-step cooking steps, and chef tips.",
        "focus": "Dish name, prep & cook time, categorized ingredients, numbered recipe steps, culinary secrets."
    },
    "shopping": {
        "name": "🛒 Shopping, Deals & Expenses",
        "description": "Itemized list, quantities, price comparisons, budget estimates, and purchase priorities.",
        "focus": "List of items, estimated costs, store deals mentioned, urgent vs optional purchases."
    },
    "general": {
        "name": "📝 General Audio Note",
        "description": "Comprehensive structured note with key takeaways, highlights, and bullet points.",
        "focus": "General overview, main discussion topics, important highlights, and follow-ups."
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# ③  Model Loaders  (GPU-aware)
# ══════════════════════════════════════════════════════════════════════════════

def get_whisper_model(model_size: str = "base"):
    """
    Lazy-load faster-whisper using the auto-detected best backend.
    GPU   → float16 CUDA (2-4× faster)
    MPS   → float16 auto
    CPU   → int8 quantized (low-RAM friendly)
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[Whisper] ⚠️ faster-whisper is NOT installed. Run 'python install.py' on the server.")
            return None

        device  = HW["whisper_device"]
        compute = HW["whisper_compute"]
        print(f"[Whisper] Loading '{model_size}' on {device.upper()} ({compute}) threads={CPU_THREADS}...")
        try:
            _WHISPER_MODEL = WhisperModel(
                model_size,
                device=device,
                compute_type=compute,
                cpu_threads=CPU_THREADS,
            )
            print(f"[Whisper] ✅ Loaded successfully on {device.upper()}!")
        except Exception as e:
            print(f"[Whisper] ⚠️ Could not load on {device} ({e}). Retrying on CPU (int8)...")
            try:
                _WHISPER_MODEL = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=CPU_THREADS,
                )
                print(f"[Whisper] ✅ Loaded successfully on CPU fallback!")
            except Exception as e2:
                print(f"[Whisper] ⚠️ Could not load on CPU fallback either: {e2}")
                return None
    return _WHISPER_MODEL


def get_llama_model():
    """
    Lazy-load Gemma 3 1B GGUF via llama-cpp-python.
    GPU (CUDA/ROCm) → n_gpu_layers=-1  (all layers offloaded, very fast)
    Apple MPS        → n_gpu_layers=1   (partial offload)
    CPU              → n_gpu_layers=0   (pure CPU int4, works on any laptop)
    """
    global _LLAMA_MODEL
    if _LLAMA_MODEL is None:
        try:
            from llama_cpp import Llama
            n_gpu = HW["llama_n_gpu_layers"]
            print(f"[Gemma] Loading {GGUF_FILENAME} — GPU layers: {n_gpu} "
                  f"({'full GPU' if n_gpu == -1 else 'CPU only' if n_gpu == 0 else 'partial GPU'})...")
            _LLAMA_MODEL = Llama.from_pretrained(
                repo_id=GGUF_REPO_ID,
                filename=GGUF_FILENAME,
                n_ctx=4096,
                n_threads=CPU_THREADS,
                n_gpu_layers=n_gpu,
                verbose=False,
            )
            backend_label = HW["backend"].upper()
            print(f"[Gemma] ✅ Loaded successfully on {backend_label}!")
        except Exception as e:
            print(f"[Gemma] ⚠️  llama_cpp not loaded: {e}")
            return None
    return _LLAMA_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# ④  Hardware Status API  (exposed to FastAPI for the UI status badge)
# ══════════════════════════════════════════════════════════════════════════════

def get_hardware_status() -> Dict[str, Any]:
    """Returns current hardware detection results for the /api/hardware endpoint."""
    return {
        "backend":         HW["backend"],
        "gpu_name":        HW["gpu_name"],
        "vram_gb":         HW["vram_gb"],
        "summary":         HW["summary"],
        "whisper_device":  HW["whisper_device"],
        "whisper_compute": HW["whisper_compute"],
        "llama_gpu_layers":HW["llama_n_gpu_layers"],
        "cpu_threads":     CPU_THREADS,
        "whisper_model":   WHISPER_MODEL_SIZE,
        "gemma_model":     GGUF_FILENAME,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ⑤  Core AI Functions  (unchanged logic, now GPU-aware via loaders above)
# ══════════════════════════════════════════════════════════════════════════════

async def transcribe_audio_file(
    file_bytes: bytes,
    filename: str,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribes audio using faster-whisper on the auto-detected best backend.
    GPU   → 2-4× real-time on float16 CUDA
    CPU   → ~1× real-time on int8 (slower but works on any machine)
    """
    temp_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_{filename}")

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    error_detail = ""
    try:
        model = get_whisper_model(WHISPER_MODEL_SIZE)
        if model is not None:
            t0         = time.time()
            lang_param = None if (not language or language == "auto") else language

            # First attempt: with VAD filtering (cleans up silences)
            try:
                segments, info = model.transcribe(temp_path, vad_filter=True, language=lang_param)
                lines = [s.text.strip() for s in segments]
            except Exception as vad_err:
                print(f"[Whisper] VAD filter attempt failed ({vad_err}), retrying without VAD...")
                segments, info = model.transcribe(temp_path, vad_filter=False, language=lang_param)
                lines = [s.text.strip() for s in segments]

            transcript = " ".join(lines).strip()
            elapsed  = round(time.time() - t0, 1)
            detected_lang = getattr(info, "language", language or "auto")
            duration = round(getattr(info, "duration", 0))
            backend_label = HW["backend"].upper()

            if transcript:
                return {
                    "success":    True,
                    "transcript": transcript,
                    "language":   detected_lang,
                    "duration":   duration,
                    "provider":   f"Whisper {WHISPER_MODEL_SIZE} on {backend_label} ({HW['whisper_compute']}) · {elapsed}s"
                }
            else:
                error_detail = "Whisper processed the file but detected no clear spoken words."
        else:
            error_detail = "faster-whisper is not installed or could not be loaded."
    except Exception as e:
        error_detail = f"Transcription failed ({e})."
        print(f"[Whisper Transcribe Error]: {e}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return {
        "success":    False,
        "transcript": f"[Audio file uploaded & saved. {error_detail} Run 'python install.py' on the server, then click '🎙️ Transcribe Audio' to extract the text, or type/edit it manually.]",
        "language":   language or "auto",
        "duration":   0,
        "provider":   "Manual Mode"
    }


def call_local_gemma(prompt: str, system_instruction: str = "", max_tokens: int = 500) -> str:
    """Runs a prompt through local Gemma 3 1B GGUF (GPU if available, else CPU)."""
    llm = get_llama_model()
    if llm is not None:
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Local Gemma Error]: {e}")

    return generate_rule_based_summary(prompt)


def generate_rule_based_summary(text: str) -> str:
    """Heuristic fallback — works before model is downloaded/cached."""
    words     = text.split()
    first_few = " ".join(words[:40]) if words else "Audio recording"
    return json.dumps({
        "category":   "general",
        "title":      "Audio Note Summary",
        "overview":   f"Audio note transcript captured: {first_few}...",
        "key_points": [
            "Audio processed locally and AES-256 encrypted in SQLite.",
            "Full transcript available for editing in the browser editor.",
            "Ask questions to EchoNote AI chatbot in the right panel."
        ],
        "action_items": [
            "Review and edit the extracted transcript in the editor tab.",
            "Ask questions to EchoNote AI Copilot in the right panel."
        ],
        "study_exam_questions": [
            {"question": "What is the primary topic of this audio?", "answer": first_few}
        ],
        "outline_mindmap": "# Note Outline\n- Introduction\n- Core Discussion\n- Action Items"
    })


def split_words(text: str, size: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


async def auto_detect_category(transcript: str) -> str:
    """Classifies audio type locally using Gemma 1B."""
    sample  = " ".join(transcript.split()[:250])
    options = ", ".join(CATEGORY_PERSONAS.keys())
    prompt  = (
        f"Read this text from an audio recording. Which type is it?\n"
        f"Choose exactly one from this list: {options}.\n"
        f"Reply with ONLY the lowercase type name.\n\nTEXT:\n{sample}"
    )
    reply = call_local_gemma(prompt, max_tokens=15).lower()
    for cat in CATEGORY_PERSONAS.keys():
        if cat in reply:
            return cat
    return "general"


async def generate_enhanced_summary(
    transcript: str,
    category: str = "auto",
    custom_instructions: str = ""
) -> Tuple[str, Dict[str, Any], str]:
    """
    Generates structured summaries for students, teachers, meetings, and professionals.
    Runs on the auto-detected best backend (GPU / MPS / CPU).
    """
    if not transcript or len(transcript.strip()) < 5:
        return "No transcript text available to summarize.", {}, "general"

    if category == "auto" or category not in CATEGORY_PERSONAS:
        category = await auto_detect_category(transcript)

    persona = CATEGORY_PERSONAS.get(category, CATEGORY_PERSONAS["general"])
    rules   = "Use clear, concise sentences. Use only the given text. Do not invent facts."

    system_prompt = f"""You are EchoNote Chief Note Architect.
Generate a structured note for {persona['name']}.
Focus: {persona['focus']}
{rules}

Output ONLY a valid JSON object matching this schema:
{{
  "category": "{category}",
  "title": "A concise title (max 7 words)",
  "overview": "Clear summary paragraph (3-6 sentences)",
  "key_points": [
    "Key point 1",
    "Key point 2",
    "Key point 3"
  ],
  "action_items": [
    "Action item or deadline 1",
    "Action item or deadline 2"
  ],
  "study_exam_questions": [
    {{
      "question": "Important exam or review question",
      "answer": "Answer based on text"
    }}
  ],
  "outline_mindmap": "Markdown outline of the main points"
}}
"""

    parts = split_words(transcript, 450)
    if len(parts) == 1:
        user_prompt = (f"TRANSCRIPT:\n{parts[0]}\n\n"
                       f"SPECIAL REQUESTS:\n{custom_instructions or 'Standard synthesis'}")
        raw_response = call_local_gemma(user_prompt, system_instruction=system_prompt, max_tokens=500)
    else:
        part_notes = []
        for p in parts[:4]:
            part_summary = call_local_gemma(
                f"Summarize this part in 3 bullet points: {persona['focus']}\n\nTEXT:\n{p}",
                max_tokens=200
            )
            part_notes.append(part_summary)
        joined       = "\n".join(part_notes)
        user_prompt  = (f"NOTES FROM AUDIO:\n{joined}\n\n"
                        f"SPECIAL REQUESTS:\n{custom_instructions or 'Standard synthesis'}")
        raw_response = call_local_gemma(user_prompt, system_instruction=system_prompt, max_tokens=500)

    # Clean markdown code fences if present
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"): cleaned = cleaned[7:]
    if cleaned.startswith("```"):     cleaned = cleaned[3:]
    if cleaned.endswith("```"):       cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        md_summary = f"""# {data.get('title', 'Audio Note')}

**Category:** {persona['name']}

### 📌 Overview
{data.get('overview', '')}

### 💡 Key Takeaways
""" + "\n".join([f"- {kp}" for kp in data.get("key_points", [])])

        if data.get("action_items"):
            md_summary += "\n\n### ✅ Action Items & Tasks\n" + \
                          "\n".join([f"- [ ] {ai}" for ai in data.get("action_items", [])])

        if data.get("study_exam_questions"):
            md_summary += "\n\n### 🧠 Study Guide & Exam Prep\n"
            for idx, q in enumerate(data.get("study_exam_questions", []), 1):
                md_summary += f"\n**Q{idx}: {q.get('question')}**\n> *Answer:* {q.get('answer')}\n"

        return md_summary, data, category
    except Exception:
        fallback_data = {
            "category": category,
            "title":    "Audio Note Summary",
            "overview": raw_response[:300],
            "key_points": ["Review the detailed transcript and notes below."],
            "action_items": [],
            "study_exam_questions": [],
            "outline_mindmap": ""
        }
        return raw_response, fallback_data, category


async def answer_transcript_question(
    question: str,
    transcript: str,
    summary: str,
    chat_history: List[Dict[str, str]],
    category: str = "general"
) -> str:
    """
    Grounded Q&A — runs on auto-detected backend (GPU / MPS / CPU).
    Uses keyword-matched transcript excerpts for accurate grounded answers.
    """
    persona = CATEGORY_PERSONAS.get(category, CATEGORY_PERSONAS["general"])

    chunks = split_words(transcript, 200)
    if len(chunks) <= 3:
        context = "\n\n".join(chunks)
    else:
        q_words = set(question.lower().split())
        scored  = [(sum(1 for w in q_words if w in c.lower()), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        context = "\n\n".join(c for _, c in scored[:3])

    prompt = (
        f"You help a person understand an audio recording ({persona['name']}).\n"
        f"Answer the question using ONLY the SUMMARY and EXCERPTS below.\n"
        f"If the answer is not in the text, say: 'I did not hear this in the audio.'\n\n"
        f"SUMMARY:\n{summary[:800]}\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    return call_local_gemma(prompt, max_tokens=350)
