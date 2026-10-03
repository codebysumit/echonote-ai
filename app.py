"""
EchoNote AI — Local Inference Studio Server (FastAPI)
Auto-detects GPU (NVIDIA CUDA / Apple MPS) and falls back to CPU.
Encrypted SQLite storage with full in-browser editor and Gemma AI intelligence.
"""

import os
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
import ai_service

app = FastAPI(title="EchoNote AI — Local Intelligence Studio", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
def on_startup():
    """
    Auto-create / Auto-connect to SQLite on every startup.
    ─────────────────────────────────────────────────────
    • First run (DB missing) → creates notes.db + all tables from scratch.
    • Subsequent runs         → connects to existing notes.db, runs
                                CREATE TABLE IF NOT EXISTS so schema is safe.
    • The uploads/ directory is also guaranteed to exist.
    """
    # Ensure required directories exist (important for fresh clones)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    is_new = database.is_new_database()

    database.init_db()   # CREATE TABLE IF NOT EXISTS — safe for both cases

    if is_new:
        print("─" * 60)
        print("  📦  DATABASE: Created new encrypted SQLite database.")
        print(f"       Path : {database.DB_PATH}")
        print("       All your notes will be AES-256 encrypted here.")
        print("       This file is excluded from Git (.gitignore).")
        print("─" * 60)
    else:
        note_count = _count_notes()
        print("─" * 60)
        print("  ✅  DATABASE: Connected to existing encrypted SQLite database.")
        print(f"       Path  : {database.DB_PATH}")
        print(f"       Notes : {note_count} encrypted note(s) loaded.")
        print("─" * 60)


def _count_notes() -> int:
    """Quick helper — returns how many notes are currently in the database."""
    try:
        import sqlite3 as _sq
        conn = _sq.connect(database.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notes")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0

class UpdateNoteRequest(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    question: str

class RegenerateSummaryRequest(BaseModel):
    category: Optional[str] = "auto"
    custom_instructions: Optional[str] = ""

class ManualNoteRequest(BaseModel):
    title: str
    category: Optional[str] = "general"
    transcript: str
    language: Optional[str] = "auto"

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Smart Audio Notes (Local CPU)</h1>")

@app.get("/api/hardware")
def get_hardware_info():
    """
    Returns current hardware detection results.
    Used by the frontend to display the GPU/CPU status badge.
    """
    return JSONResponse(content=ai_service.get_hardware_status())


@app.get("/api/categories")
async def get_categories():
    return ai_service.CATEGORY_PERSONAS

@app.get("/api/notes")
async def list_all_notes(
    q: str = Query("", description="Search term"),
    category: str = Query("all", description="Category filter"),
    x_encryption_key: Optional[str] = Header(None)
):
    notes = database.list_notes(search_query=q, category=category, custom_key=x_encryption_key)
    return {"notes": notes}

@app.get("/api/notes/{note_id}")
async def get_note(note_id: str, x_encryption_key: Optional[str] = Header(None)):
    note = database.get_note_by_id(note_id, custom_key=x_encryption_key)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

@app.post("/api/upload-audio")
async def upload_and_process_audio(
    file: UploadFile = File(...),
    category: str = Form("auto"),
    language: str = Form("auto"),
    x_encryption_key: Optional[str] = Header(None)
):
    note_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
    saved_filename = f"{note_id}{ext}"
    saved_filepath = os.path.join(UPLOAD_DIR, saved_filename)

    contents = await file.read()
    with open(saved_filepath, "wb") as f:
        f.write(contents)

    # 1. Local CPU Whisper Transcription
    transcription_result = await ai_service.transcribe_audio_file(
        file_bytes=contents,
        filename=saved_filename,
        language=language if language != "auto" else None
    )

    transcript = transcription_result.get("transcript", "")
    detected_lang = transcription_result.get("language", language or "auto")
    audio_duration = transcription_result.get("duration", 0)

    # 2. Local CPU Gemma Multi-Persona Summarization
    md_summary, struct_data, detected_category = await ai_service.generate_enhanced_summary(
        transcript=transcript,
        category=category,
        custom_instructions=""
    )

    title = struct_data.get("title") or (file.filename.rsplit(".", 1)[0] if file.filename else "Audio Note")

    # 3. Encrypted SQLite Storage
    saved_note = database.save_note(
        note_id=note_id,
        title=title,
        category=detected_category,
        audio_filename=saved_filename,
        audio_duration=audio_duration,
        language=detected_lang,
        transcript=transcript,
        summary=md_summary,
        structured_data=struct_data,
        custom_key=x_encryption_key
    )

    return {
        "success": True,
        "note": saved_note,
        "transcription_provider": transcription_result.get("provider", "Local CPU Whisper")
    }

@app.post("/api/notes/create-manual")
async def create_manual_note(
    req: ManualNoteRequest,
    x_encryption_key: Optional[str] = Header(None)
):
    note_id = str(uuid.uuid4())
    md_summary, struct_data, detected_category = await ai_service.generate_enhanced_summary(
        transcript=req.transcript,
        category=req.category or "general",
        custom_instructions=""
    )

    saved_note = database.save_note(
        note_id=note_id,
        title=req.title or struct_data.get("title", "New Audio Note"),
        category=detected_category,
        audio_filename="",
        audio_duration=0,
        language=req.language or "auto",
        transcript=req.transcript,
        summary=md_summary,
        structured_data=struct_data,
        custom_key=x_encryption_key
    )
    return {"success": True, "note": saved_note}

@app.put("/api/notes/{note_id}")
async def update_note(
    note_id: str,
    req: UpdateNoteRequest,
    x_encryption_key: Optional[str] = Header(None)
):
    existing = database.get_note_by_id(note_id, custom_key=x_encryption_key)
    if not existing:
        raise HTTPException(status_code=404, detail="Note not found")

    updated = database.update_note_transcript_summary(
        note_id=note_id,
        title=req.title,
        category=req.category,
        transcript=req.transcript,
        summary=req.summary,
        structured_data=req.structured_data,
        custom_key=x_encryption_key
    )
    return {"success": True, "note": updated}

@app.post("/api/notes/{note_id}/regenerate-summary")
async def regenerate_summary(
    note_id: str,
    req: RegenerateSummaryRequest,
    x_encryption_key: Optional[str] = Header(None)
):
    note = database.get_note_by_id(note_id, custom_key=x_encryption_key)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    md_summary, struct_data, detected_category = await ai_service.generate_enhanced_summary(
        transcript=note["transcript"],
        category=req.category or note["category"],
        custom_instructions=req.custom_instructions or ""
    )

    updated = database.update_note_transcript_summary(
        note_id=note_id,
        category=detected_category,
        summary=md_summary,
        structured_data=struct_data,
        custom_key=x_encryption_key
    )
    return {"success": True, "note": updated}

@app.post("/api/notes/{note_id}/chat")
async def chat_with_note(
    note_id: str,
    req: ChatRequest,
    x_encryption_key: Optional[str] = Header(None)
):
    note = database.get_note_by_id(note_id, custom_key=x_encryption_key)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    user_msg_id = str(uuid.uuid4())
    database.save_chat_message(
        note_id=note_id,
        message_id=user_msg_id,
        role="user",
        content=req.question,
        custom_key=x_encryption_key
    )

    assistant_reply = await ai_service.answer_transcript_question(
        question=req.question,
        transcript=note["transcript"],
        summary=note["summary"],
        chat_history=note.get("chats", []),
        category=note.get("category", "general")
    )

    ai_msg_id = str(uuid.uuid4())
    saved_ai_msg = database.save_chat_message(
        note_id=note_id,
        message_id=ai_msg_id,
        role="assistant",
        content=assistant_reply,
        custom_key=x_encryption_key
    )

    return {
        "success": True,
        "user_message": {"id": user_msg_id, "role": "user", "content": req.question},
        "assistant_message": saved_ai_msg
    }

@app.delete("/api/notes/{note_id}")
async def delete_note_endpoint(note_id: str):
    note = database.get_note_by_id(note_id)
    if note and note.get("audio_filename"):
        audio_file = os.path.join(UPLOAD_DIR, note["audio_filename"])
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception:
                pass

    success = database.delete_note(note_id)
    return {"success": success}

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path)

@app.get("/api/notes/{note_id}/export/{format_type}")
async def export_note(
    note_id: str,
    format_type: str,
    x_encryption_key: Optional[str] = Header(None)
):
    note = database.get_note_by_id(note_id, custom_key=x_encryption_key)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    title = note["title"]
    category = note["category"]
    transcript = note["transcript"]
    summary = note["summary"]
    created_at = note["created_at"]

    chats_text = ""
    for c in note.get("chats", []):
        chats_text += f"\n**{c['role'].title()}**: {c['content']}\n"

    if format_type in ("markdown", "md"):
        content = f"""# {title}
**Category:** {category.upper()} | **Date:** {created_at}

---

## 📋 Executive Summary
{summary}

---

## 🎙️ Full Transcript
{transcript}

---

## 💬 AI Q&A Log
{chats_text if chats_text else "No questions asked."}
"""
        return PlainTextResponse(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{title.replace(" ", "_")}_notes.md"'}
        )

    elif format_type == "json":
        return JSONResponse(
            content=note,
            headers={"Content-Disposition": f'attachment; filename="{title.replace(" ", "_")}_notes.json"'}
        )

    else:
        txt_content = f"{title}\nCategory: {category}\nDate: {created_at}\n\nSUMMARY:\n{summary}\n\nTRANSCRIPT:\n{transcript}"
        return PlainTextResponse(
            content=txt_content,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{title.replace(" ", "_")}_notes.txt"'}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Smart Audio Notes (100% Local CPU) on http://127.0.0.1:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
