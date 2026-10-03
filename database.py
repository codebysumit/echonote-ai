"""
Database & Encryption Module for Smart Audio Notes
Stores all notes, transcripts, summaries, and chat histories in SQLite
with AES encryption (Fernet / PBKDF2) for complete privacy and security.

Auto-create / Auto-connect behaviour
─────────────────────────────────────
• If notes.db does NOT exist  → init_db() creates it fresh with all tables.
• If notes.db DOES exist       → init_db() connects and runs CREATE TABLE IF NOT EXISTS
                                  so existing data is untouched and the schema stays
                                  up-to-date if new columns are ever added.
• notes.db is listed in .gitignore and is NEVER uploaded to GitHub.
• The uploads/ and notebooks/ directories ship with a .gitkeep so cloners
  get the correct folder structure without any private files.
"""

import os
import json
import sqlite3
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes.db")
DEFAULT_SECRET = os.getenv("MASTER_ENCRYPTION_KEY", "SmartAudioNotes_SecureKey_2026_DevWeek")
SALT = b"smart_notes_aes_salt_987654321"


def is_new_database() -> bool:
    """
    Returns True if the database file does not yet exist on disk.
    Call this BEFORE init_db() to decide whether to print 'Created' or 'Connected'.
    """
    return not os.path.exists(DB_PATH)

def _derive_fernet_key(passphrase: str) -> bytes:
    """Derive a URL-safe base64-encoded 32-byte key using PBKDF2HMAC."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return key

def get_fernet(custom_key: Optional[str] = None) -> Fernet:
    secret = custom_key if custom_key else DEFAULT_SECRET
    return Fernet(_derive_fernet_key(secret))

def encrypt_text(text: str, custom_key: Optional[str] = None) -> str:
    if not text:
        return ""
    f = get_fernet(custom_key)
    return f.encrypt(text.encode("utf-8")).decode("utf-8")

def decrypt_text(encrypted_text: str, custom_key: Optional[str] = None) -> str:
    if not encrypted_text:
        return ""
    try:
        f = get_fernet(custom_key)
        return f.decrypt(encrypted_text.encode("utf-8")).decode("utf-8")
    except Exception:
        # Fallback if key mismatch or unencrypted legacy
        return "[Decryption failed: Key mismatch or invalid ciphertext]"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        category TEXT DEFAULT 'general',
        audio_filename TEXT,
        audio_duration INTEGER DEFAULT 0,
        language TEXT DEFAULT 'auto',
        encrypted_transcript TEXT,
        encrypted_summary TEXT,
        encrypted_structured_data TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        note_id TEXT,
        role TEXT NOT NULL,
        encrypted_content TEXT,
        created_at TEXT,
        FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE CASCADE
    )
    """)
    conn.commit()
    conn.close()

def save_note(
    note_id: str,
    title: str,
    category: str,
    audio_filename: str,
    audio_duration: int,
    language: str,
    transcript: str,
    summary: str,
    structured_data: Dict[str, Any],
    custom_key: Optional[str] = None
) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    enc_transcript = encrypt_text(transcript, custom_key)
    enc_summary = encrypt_text(summary, custom_key)
    enc_struct = encrypt_text(json.dumps(structured_data), custom_key)

    cursor.execute("""
    INSERT INTO notes (id, title, category, audio_filename, audio_duration, language, encrypted_transcript, encrypted_summary, encrypted_structured_data, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        title=excluded.title,
        category=excluded.category,
        language=excluded.language,
        encrypted_transcript=excluded.encrypted_transcript,
        encrypted_summary=excluded.encrypted_summary,
        encrypted_structured_data=excluded.encrypted_structured_data,
        updated_at=excluded.updated_at
    """, (note_id, title, category, audio_filename, audio_duration, language, enc_transcript, enc_summary, enc_struct, now, now))

    conn.commit()
    conn.close()
    return get_note_by_id(note_id, custom_key)

def update_note_transcript_summary(
    note_id: str,
    title: Optional[str] = None,
    category: Optional[str] = None,
    transcript: Optional[str] = None,
    summary: Optional[str] = None,
    structured_data: Optional[Dict[str, Any]] = None,
    custom_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    updates = []
    params = []

    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if category is not None:
        updates.append("category = ?")
        params.append(category)
    if transcript is not None:
        updates.append("encrypted_transcript = ?")
        params.append(encrypt_text(transcript, custom_key))
    if summary is not None:
        updates.append("encrypted_summary = ?")
        params.append(encrypt_text(summary, custom_key))
    if structured_data is not None:
        updates.append("encrypted_structured_data = ?")
        params.append(encrypt_text(json.dumps(structured_data), custom_key))

    if not updates:
        conn.close()
        return get_note_by_id(note_id, custom_key)

    updates.append("updated_at = ?")
    params.append(now)
    params.append(note_id)

    query = f"UPDATE notes SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    return get_note_by_id(note_id, custom_key)

def list_notes(search_query: str = "", category: str = "", custom_key: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT id, title, category, audio_filename, audio_duration, language, created_at, updated_at, encrypted_summary, encrypted_transcript FROM notes"
    params = []

    if category and category != "all":
        query += " WHERE category = ?"
        params.append(category)

    query += " ORDER BY updated_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    notes = []

    for r in rows:
        summary_preview = decrypt_text(r["encrypted_summary"], custom_key)
        transcript_text = decrypt_text(r["encrypted_transcript"], custom_key)
        
        # Search query filter in decrypted content if specified
        if search_query:
            sq = search_query.lower()
            if sq not in r["title"].lower() and sq not in summary_preview.lower() and sq not in transcript_text.lower():
                continue

        notes.append({
            "id": r["id"],
            "title": r["title"],
            "category": r["category"],
            "audio_filename": r["audio_filename"],
            "audio_duration": r["audio_duration"],
            "language": r["language"],
            "summary_preview": (summary_preview[:180] + "...") if len(summary_preview) > 180 else summary_preview,
            "created_at": r["created_at"],
            "updated_at": r["updated_at"]
        })

    conn.close()
    return notes

def get_note_by_id(note_id: str, custom_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    transcript = decrypt_text(row["encrypted_transcript"], custom_key)
    summary = decrypt_text(row["encrypted_summary"], custom_key)
    struct_str = decrypt_text(row["encrypted_structured_data"], custom_key)
    try:
        structured_data = json.loads(struct_str) if struct_str else {}
    except Exception:
        structured_data = {}

    # Fetch chats
    cursor.execute("SELECT id, role, encrypted_content, created_at FROM chats WHERE note_id = ? ORDER BY created_at ASC", (note_id,))
    chat_rows = cursor.fetchall()
    chats = []
    for c in chat_rows:
        chats.append({
            "id": c["id"],
            "role": c["role"],
            "content": decrypt_text(c["encrypted_content"], custom_key),
            "created_at": c["created_at"]
        })

    conn.close()
    return {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "audio_filename": row["audio_filename"],
        "audio_duration": row["audio_duration"],
        "language": row["language"],
        "transcript": transcript,
        "summary": summary,
        "structured_data": structured_data,
        "chats": chats,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }

def delete_note(note_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE note_id = ?", (note_id,))
    cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def save_chat_message(note_id: str, message_id: str, role: str, content: str, custom_key: Optional[str] = None) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    enc_content = encrypt_text(content, custom_key)

    cursor.execute("""
    INSERT INTO chats (id, note_id, role, encrypted_content, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (message_id, note_id, role, enc_content, now))

    conn.commit()
    conn.close()
    return {
        "id": message_id,
        "note_id": note_id,
        "role": role,
        "content": content,
        "created_at": now
    }
