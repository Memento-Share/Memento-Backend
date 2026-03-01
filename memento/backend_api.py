# backend_api.py
#
# Option A flow (record -> upload whole file -> store in Supabase -> then process):
#  - POST /recordings: Expo uploads audio file -> backend uploads to Supabase Storage "recordings"
#                      -> updates convos.media_id + media.recordings
#                      -> (optional) appends a user stub to messages.threads via RPC append_turns
#  - POST /process/image: fetch photo bytes via public URL -> Gemini -> append AI turn
#  - POST /process/audio: fetch recording bytes via public URL -> Gemini -> append AI turn
#
# Run:
#   pip install fastapi uvicorn requests pydantic
#   export SUPABASE_URL="https://dspaaqbthcdwtxfflovn.supabase.co"
#   export SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
#   export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
#   python3 -m uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000

import os
import mimetypes
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from agent_graph import init_audio_state, init_photo_state, MementoState, run_agent
from langchain.messages import AIMessage

# ---------------------------
# Config
# ---------------------------
SUPABASE_URL = "https://dspaaqbthcdwtxfflovn.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzcGFhcWJ0aGNkd3R4ZmZsb3ZuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MjMxODY1NCwiZXhwIjoyMDg3ODk0NjU0fQ.Nq3ICA7nvS1C43q1QKzRlMLmOZ8BtOdtOmk0iPh83XA"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Conversation memory (MVP global; replace with per-user/per-convo state later)
messages: list = []
follow_ups = 0

# If your buckets are PUBLIC, object URL format is:
#   {SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}
def public_object_url(bucket: str, path: str) -> str:
    path = path.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"


# ---------------------------
# Supabase helpers
# ---------------------------
def _sb_headers(prefer_return: bool = False) -> Dict[str, str]:
    h = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
    if prefer_return:
        h["Prefer"] = "return=representation"
    return h


def supabase_rpc(fn: str, payload: dict) -> Any:
    url = f"{SUPABASE_URL}/rest/v1/rpc/{fn}"
    headers = {
        **_sb_headers(),
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    try:
        return r.json()
    except Exception:
        return r.text


def supabase_get(path: str) -> Any:
    """GET helper for PostgREST paths like '/rest/v1/convos?...'"""
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=_sb_headers(), timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


def supabase_insert(table: str, payload: dict, select: str = "*") -> dict:
    """Insert into a table and return inserted row."""
    url = f"{SUPABASE_URL}/rest/v1/{table}?select={select}"
    headers = {**_sb_headers(prefer_return=True), "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    rows = r.json()
    if not rows:
        raise HTTPException(status_code=500, detail=f"Insert into {table} returned no row")
    return rows[0]


def supabase_update(table: str, match_query: str, payload: dict, select: str = "*") -> dict:
    """Patch rows in a table and return updated row. match_query like 'id=eq.123'"""
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_query}&select={select}"
    headers = {**_sb_headers(prefer_return=True), "Content-Type": "application/json"}
    r = requests.patch(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    rows = r.json()
    if not rows:
        raise HTTPException(status_code=500, detail=f"Update {table} returned no row")
    return rows[0]


# ---------------------------
# Storage upload (recordings)
# ---------------------------
def storage_upload(bucket: str, object_path: str, data: bytes, content_type: str) -> None:
    """
    Upload raw bytes to Supabase Storage.
    Uses service role key (server-only).
    """
    object_path = object_path.lstrip("/")
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{object_path}"

    headers = {
        **_sb_headers(),
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    # Try POST then PUT (some setups differ)
    r = requests.post(url, headers=headers, data=data, timeout=60)
    if r.ok:
        return

    r2 = requests.put(url, headers=headers, data=data, timeout=60)
    if r2.ok:
        return

    raise HTTPException(
        status_code=502,
        detail=f"Storage upload failed. POST: {r.status_code} {r.text} | PUT: {r2.status_code} {r2.text}",
    )


# ---------------------------
# Fetch media from public URL (for Gemini)
# ---------------------------
def fetch_media(url: str) -> Tuple[bytes, str]:
    """
    Fetch bytes from a URL and also return a MIME type.
    Primary: HTTP Content-Type header
    Fallback: mimetypes.guess_type(url)
    Final fallback: application/octet-stream
    """
    r = requests.get(url, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=f"Failed to fetch media: {r.text}")

    mime = r.headers.get("Content-Type")
    if mime:
        mime = mime.split(";")[0].strip()

    if not mime or mime == "application/octet-stream":
        mime = mimetypes.guess_type(url)[0] or "application/octet-stream"

    return r.content, mime


# ---------------------------
# Gemini calls (your LangGraph agent)
# ---------------------------
def gemini_analyze_image(image_bytes: bytes, image_mime: str) -> str:
    global messages
    state: MementoState = init_photo_state(image_bytes, image_mime)
    response = run_agent(state)
    messages = response["messages"]
    if len(messages) > 0:
        return messages[0].content
    return "Error: no response from Gemini"


def gemini_analyze_audio(audio_bytes: bytes, audio_mime: str) -> str:
    global messages, follow_ups
    state: MementoState = init_audio_state(audio_bytes, audio_mime, messages)
    response = run_agent(state)
    messages = response["messages"]
    if len(messages) > 0 and isinstance(messages[-1], AIMessage):
        return messages[-1].content
    return "Error: no response from Gemini"


# ---------------------------
# Thread schema
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


def append_ai_turn(convo_id: int, ai_text: str, kind: Kind, meta: Optional[dict] = None) -> Any:
    return append_turns(convo_id, [ThreadTurn(role="ai", text=ai_text, kind=kind, meta=meta)])


# ---------------------------
# DB helpers for convo/media
# ---------------------------
def get_convo(convo_id: int) -> dict:
    rows = supabase_get(f"/rest/v1/convos?id=eq.{convo_id}&select=id,media_id,owner_user_id")
    if not rows:
        raise HTTPException(status_code=404, detail=f"convo_id {convo_id} not found")
    return rows[0]


def ensure_media_for_convo(convo_id: int) -> int:
    convo = get_convo(convo_id)
    if convo.get("media_id") is not None:
        return int(convo["media_id"])

    media = supabase_insert("media", {}, select="id")
    media_id = int(media["id"])
    supabase_update("convos", f"id=eq.{convo_id}", {"media_id": media_id}, select="id,media_id")
    return media_id


# ---------------------------
# API models
# ---------------------------
class ProcessImageRequest(BaseModel):
    convo_id: int
    photo_path: str  # e.g. "1/uuid.jpg"


class ProcessAudioRequest(BaseModel):
    convo_id: int
    recording_path: str  # e.g. "1/uuid.m4a"


class ProcessResponse(BaseModel):
    ok: bool
    ai_text: str
    ai_mime: str
    media_url: str
    updated_message_row: Any


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Memento Python Backend")


# ---------------------------
# NEW: Upload recording from Expo -> Supabase Storage -> DB update
# ---------------------------
@app.post("/recordings")
async def upload_recording(
    convo_id: int = Form(...),
    owner_user_id: int = Form(...),
    audio: UploadFile = File(...),
    append_user_stub: bool = Form(True),
):
    """
    Expo uploads a full audio file (m4a/mp3/wav/etc).
    Backend stores it in Storage bucket: recordings
    and sets media.recordings = object_path for this convo.
    """

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    mime = audio.content_type or "application/octet-stream"
    ai_text = gemini_analyze_audio(audio_bytes, audio_mime=mime)

    # preserve extension if provided; default to .m4a
    ext = ".m4a"
    if audio.filename and "." in audio.filename:
        ext = "." + audio.filename.split(".")[-1].lower()

    object_path = f"{owner_user_id}/{uuid.uuid4().hex}{ext}"

    # 1) Upload to Supabase Storage recordings bucket
    storage_upload("recordings", object_path, audio_bytes, mime)

    # 2) Ensure media row exists, attach path
    media_id = ensure_media_for_convo(convo_id)
    supabase_update("media", f"id=eq.{media_id}", {"recordings": object_path}, select="id,recordings")

    # 3) Optionally append a user stub turn
    thread_update = None
    if append_user_stub:
        thread_update = append_turns(
            convo_id,
            [ThreadTurn(role="user", text="[voice message recorded]", kind="audio",
                        meta={"recording_path": object_path, "mime": mime})],
        )
    updated = append_ai_turn(
        convo_id=convo_id,
        ai_text=ai_text,
        kind="audio"
    )

    return {
        "ok": True,
        "convo_id": convo_id,
        "media_id": media_id,
        "recording_path": object_path,
        "ai_text": ai_text,
        # If recordings bucket is public, frontend can play this directly:
        "recording_url": public_object_url("recordings", object_path),
        "mime": mime,
        "bytes_len": len(audio_bytes),
        "thread_update": thread_update,
    }


@app.post("/process/image", response_model=ProcessResponse)
def process_image(req: ProcessImageRequest):
    url = public_object_url("photos", req.photo_path)
    img_bytes, img_mime = fetch_media(url)
    ai_text = gemini_analyze_image(img_bytes, img_mime)

    updated = append_ai_turn(
        convo_id=req.convo_id,
        ai_text=ai_text,
        kind="image",
        meta={"photo_url": url, "photo_path": req.photo_path, "mime": img_mime},
    )

    return ProcessResponse(ok=True, ai_text=ai_text, ai_mime=img_mime, media_url=url, updated_message_row=updated)


# @app.post("/process/audio", response_model=ProcessResponse)
# def process_audio(req: ProcessAudioRequest):
#     # recording_path should come from /recordings response (and be stored in media.recordings)
#     url = public_object_url("recordings", req.recording_path)
#     audio_bytes, audio_mime = fetch_media(url)

#     ai_text = gemini_analyze_audio(audio_bytes, audio_mime=audio_mime)

#     updated = append_ai_turn(
#         convo_id=req.convo_id,
#         ai_text=ai_text,
#         kind="audio",
#         meta={"recording_url": url, "recording_path": req.recording_path, "mime": audio_mime},
#     )

#     return ProcessResponse(ok=True, ai_text=ai_text, ai_mime=audio_mime, media_url=url, updated_message_row=updated)