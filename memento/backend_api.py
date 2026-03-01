# backend_api.py
#
# Python-backend-only API for:
#  - Fetching public Supabase Storage objects (photos/recordings)
#  - Extracting MIME type from HTTP headers (with fallbacks)
#  - Calling Gemini with (bytes + mime)
#  - Appending Gemini turns into Supabase messages.threads via RPC append_turns
#
# Run:
#   pip install fastapi uvicorn requests pydantic
#   export SUPABASE_URL="https://dspaaqbthcdwtxfflovn.supabase.co"
#   export SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
#   export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
#   uvicorn backend_api:app --reload --port 8000

import os
import mimetypes
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------
# Config
# ---------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# If your buckets are PUBLIC, object URL format is:
#   {SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}
def public_object_url(bucket: str, path: str) -> str:
    path = path.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"


# ---------------------------
# Supabase helpers
# ---------------------------
def supabase_rpc(fn: str, payload: dict) -> Any:
    url = f"{SUPABASE_URL}/rest/v1/rpc/{fn}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    try:
        return r.json()
    except Exception:
        return r.text


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
    # Sometimes servers return Content-Type with charset for text; for media it’s usually fine.
    if mime:
        mime = mime.split(";")[0].strip()

    if not mime or mime == "application/octet-stream":
        mime = mimetypes.guess_type(url)[0] or "application/octet-stream"

    return r.content, mime


# ---------------------------
# Gemini calls (replace internals with your SDK)
# ---------------------------
def gemini_analyze_image(image_bytes: bytes, image_mime: str, prompt: str) -> str:
    """
    Return Gemini's text response for the given image + prompt.

    Replace internals with your Gemini SDK usage.
    You will typically pass the bytes plus mime type to Gemini as a part.
    """
    # ---- Replace with real Gemini call ----
    # Example (older SDK, conceptually):
    # import google.generativeai as genai
    # genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel("gemini-1.5-flash")
    # resp = model.generate_content([
    #     prompt,
    #     {"mime_type": image_mime, "data": image_bytes}
    # ])
    # return resp.text

    return f"AI: (image processed: {image_mime}) Tell me what’s happening in this photo and why it matters to you."


def gemini_analyze_audio(audio_bytes: bytes, audio_mime: str, prompt: str) -> str:
    """
    Return Gemini's text response for the given audio + prompt.

    Replace internals with your Gemini SDK usage.
    Many teams do STT first; but if you’re using Gemini multimodal audio input,
    you’ll send bytes + mime similarly.
    """
    # ---- Replace with real Gemini call ----
    return f"AI: (audio processed: {audio_mime}) I heard you—what happened right before you noticed this?"


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


def append_ai_turn(convo_id: int, ai_text: str, kind: Kind, meta: Optional[dict] = None) -> Any:
    payload = {
        "p_convo_id": convo_id,
        "p_new_turns": [ThreadTurn(role="ai", text=ai_text, kind=kind, meta=meta).model_dump()],
    }
    return supabase_rpc("append_turns", payload)


# ---------------------------
# API models
# ---------------------------
class ProcessImageRequest(BaseModel):
    convo_id: int
    photo_path: str  # e.g. "1/uuid.jpg"
    prompt: Optional[str] = None


class ProcessAudioRequest(BaseModel):
    convo_id: int
    recording_path: str  # e.g. "1/uuid.webm"
    prompt: Optional[str] = None


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


@app.post("/process/image", response_model=ProcessResponse)
def process_image(req: ProcessImageRequest):
    # 1) Build public URL
    url = public_object_url("photos", req.photo_path)

    # 2) Fetch bytes + mime
    img_bytes, img_mime = fetch_media(url)

    # 3) Gemini
    prompt = req.prompt or (
        "You are talking to an elderly user about a cherished photo or item. "
        "Ask one warm, simple follow-up question."
    )
    ai_text = gemini_analyze_image(img_bytes, img_mime, prompt)

    # 4) Append to DB thread
    updated = append_ai_turn(
        convo_id=req.convo_id,
        ai_text=ai_text,
        kind="image",
        meta={"photo_url": url, "photo_path": req.photo_path, "mime": img_mime},
    )

    # 5) Return for frontend
    return ProcessResponse(ok=True, ai_text=ai_text, ai_mime=img_mime, media_url=url, updated_message_row=updated)


@app.post("/process/audio", response_model=ProcessResponse)
def process_audio(req: ProcessAudioRequest):
    # 1) Build public URL
    url = public_object_url("recordings", req.recording_path)

    # 2) Fetch bytes + mime
    audio_bytes, audio_mime = fetch_media(url)

    # 3) Gemini
    prompt = req.prompt or (
        "You are helping an elderly user talk about what they said. "
        "Extract key details, then ask one short follow-up question."
    )
    ai_text = gemini_analyze_audio(audio_bytes, audio_mime, prompt)

    # 4) Append to DB thread
    updated = append_ai_turn(
        convo_id=req.convo_id,
        ai_text=ai_text,
        kind="audio",
        meta={"recording_url": url, "recording_path": req.recording_path, "mime": audio_mime},
    )

    # 5) Return
    return ProcessResponse(ok=True, ai_text=ai_text, ai_mime=audio_mime, media_url=url, updated_message_row=updated)