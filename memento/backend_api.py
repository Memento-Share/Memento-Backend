# backend_api.py
#
# Backend-only API:
#  - POST /recordings: store final MP3 in Supabase Storage recordings bucket + update DB
#  - POST /gemini/response: append Gemini text to messages.threads via RPC append_turns
#
# Env:
#   SUPABASE_URL="https://dspaaqbthcdwtxfflovn.supabase.co"
#   SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"   # server-only
#
# Run:
#   pip install fastapi uvicorn requests pydantic
#   uvicorn backend_api:app --reload --port 8000

import os
import uuid
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

app = FastAPI(title="Memento Backend (store MP3 + append Gemini)")

# ---------------------------
# Supabase HTTP helpers
# ---------------------------
def _auth_headers_json(prefer_return: bool = False) -> Dict[str, str]:
    h = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer_return:
        h["Prefer"] = "return=representation"
    return h

def supabase_get(path: str) -> Any:
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=_auth_headers_json(), timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

def supabase_insert(table: str, payload: dict, select: str = "*") -> dict:
    url = f"{SUPABASE_URL}/rest/v1/{table}?select={select}"
    r = requests.post(url, headers=_auth_headers_json(prefer_return=True), json=payload, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    rows = r.json()
    if not rows:
        raise HTTPException(status_code=500, detail=f"Insert into {table} returned no row")
    return rows[0]

def supabase_update(table: str, match_query: str, payload: dict, select: str = "*") -> dict:
    # match_query example: "id=eq.123"
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_query}&select={select}"
    r = requests.patch(url, headers=_auth_headers_json(prefer_return=True), json=payload, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    rows = r.json()
    if not rows:
        raise HTTPException(status_code=500, detail=f"Update {table} returned no row")
    return rows[0]

def supabase_rpc(fn: str, payload: dict) -> Any:
    url = f"{SUPABASE_URL}/rest/v1/rpc/{fn}"
    r = requests.post(url, headers=_auth_headers_json(), json=payload, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    try:
        return r.json()
    except Exception:
        return r.text

# ---------------------------
# Storage upload (recordings bucket)
# ---------------------------
def storage_upload(bucket: str, object_path: str, data: bytes, content_type: str) -> None:
    """
    Upload raw bytes to Supabase Storage.
    Works for private/public buckets.
    Tries POST then PUT because some environments differ.
    """
    object_path = object_path.lstrip("/")
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{object_path}"

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",  # allow overwrite if same path (optional)
    }

    # Try POST
    r = requests.post(url, headers=headers, data=data, timeout=60)
    if r.ok:
        return

    # Fallback: Try PUT
    r2 = requests.put(url, headers=headers, data=data, timeout=60)
    if r2.ok:
        return

    raise HTTPException(
        status_code=502,
        detail=f"Storage upload failed. POST: {r.status_code} {r.text} | PUT: {r2.status_code} {r2.text}",
    )

def storage_public_url(bucket: str, object_path: str) -> str:
    object_path = object_path.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{object_path}"

# ---------------------------
# Thread schema + append helper
# ---------------------------
Role = Literal["user", "ai"]
Kind = Literal["chat", "voice", "image", "audio"]

class ThreadTurn(BaseModel):
    role: Role
    text: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    kind: Optional[Kind] = None
    meta: Optional[Dict[str, Any]] = None

def append_turns(convo_id: int, turns: List[ThreadTurn]) -> Any:
    payload = {
        "p_convo_id": convo_id,
        "p_new_turns": [t.model_dump() for t in turns],
    }
    return supabase_rpc("append_turns", payload)

# ---------------------------
# DB helpers
# ---------------------------
def get_convo(convo_id: int) -> dict:
    rows = supabase_get(f"/rest/v1/convos?id=eq.{convo_id}&select=id,media_id,owner_user_id")
    if not rows:
        raise HTTPException(status_code=404, detail=f"convo_id {convo_id} not found")
    return rows[0]

def ensure_media_for_convo(convo_id: int) -> int:
    convo = get_convo(convo_id)
    media_id = convo.get("media_id")
    if media_id is not None:
        return int(media_id)

    media = supabase_insert("media", {}, select="id")
    media_id = int(media["id"])
    supabase_update("convos", f"id=eq.{convo_id}", {"media_id": media_id}, select="id,media_id")
    return media_id

# ---------------------------
# API 1: store MP3 -> recordings bucket + update DB
# ---------------------------
@app.post("/recordings")
async def post_recording(
    convo_id: int = Form(...),
    owner_user_id: int = Form(...),
    audio: UploadFile = File(...),
    append_user_stub: bool = Form(True),
):
    """
    Receives an MP3 (or other audio) file and stores it in Supabase Storage 'recordings' bucket.
    Updates:
      - ensures convos.media_id
      - sets media.recordings to storage path
    Optionally appends a '[voice message recorded]' user stub to messages.threads
    """

    convo = get_convo(convo_id)

    # Optional ownership check
    if int(convo["owner_user_id"]) != int(owner_user_id):
        raise HTTPException(status_code=403, detail="owner_user_id does not match convo owner")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    # prefer UploadFile content_type; for mp3 it should be audio/mpeg
    mime = audio.content_type or "audio/mpeg"

    # Create a path; keep it unguessable
    ext = ".mp3"
    if audio.filename and "." in audio.filename:
        # keep original extension if present (e.g. .webm/.wav)
        ext = "." + audio.filename.split(".")[-1].lower()

    object_path = f"{owner_user_id}/{uuid.uuid4().hex}{ext}"

    # Upload to Storage bucket
    storage_upload(bucket="recordings", object_path=object_path, data=audio_bytes, content_type=mime)

    # Ensure media row exists, then set recordings path
    media_id = ensure_media_for_convo(convo_id)
    updated_media = supabase_update(
        "media",
        f"id=eq.{media_id}",
        {"recordings": object_path},
        select="id,recordings,created_at",
    )

    thread_update = None
    if append_user_stub:
        thread_update = append_turns(
            convo_id,
            [
                ThreadTurn(
                    role="user",
                    text="[voice message recorded]",
                    kind="audio",
                    meta={"media_id": media_id, "recording_path": object_path, "mime": mime},
                )
            ],
        )

    return {
        "ok": True,
        "convo_id": convo_id,
        "media_id": media_id,
        "recording_path": object_path,
        "recording_public_url": storage_public_url("recordings", object_path),
        "mime": mime,
        "bytes_len": len(audio_bytes),
        "media_update": updated_media,
        "thread_update": thread_update,
    }

# ---------------------------
# API 2: post Gemini response later
# ---------------------------
class GeminiResponseRequest(BaseModel):
    convo_id: int
    ai_text: str
    kind: Optional[Literal["audio", "image", "chat"]] = "audio"
    meta: Optional[Dict[str, Any]] = None

@app.post("/gemini/response")
def post_gemini_response(req: GeminiResponseRequest):
    """
    Appends Gemini's response to messages.threads for the convo.
    """
    updated = append_turns(
        req.convo_id,
        [ThreadTurn(role="ai", text=req.ai_text, kind=req.kind, meta=req.meta)],
    )
    return {"ok": True, "convo_id": req.convo_id, "updated_message_row": updated}