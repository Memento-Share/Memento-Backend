import os
from typing import Any, Dict, List, Literal, Optional
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
# {SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}
def public_object_url(bucket: str, path: str) -> str:
    path = path.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"

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

def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=f"Failed to fetch media: {r.text}")
    return r.content

# ---------------------------
# Gemini calls (implementable)
# ---------------------------
# NOTE: I’m giving you a clean wrapper. Depending on your Gemini SDK version,
# the exact call differs slightly. This keeps the boundary clear.
#
# If you’re using google-genai (newer) or google.generativeai (older),
# swap the internals but keep the same function signatures.

def gemini_analyze_image(image_bytes: bytes, prompt: str) -> str:
    """
    Return Gemini's text response for the given image + prompt.
    Replace internals with your Gemini SDK usage.
    """
    # ---- PSEUDO IMPLEMENTATION HOOK ----
    # Example (older SDK):
    # import google.generativeai as genai
    # genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel("gemini-1.5-flash")
    # resp = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
    # return resp.text
    #
    # For now, return placeholder:
    return "AI: (image processed) Tell me more about what’s in this photo and why it matters to you."

def gemini_analyze_audio(audio_bytes: bytes, prompt: str) -> str:
    """
    Return Gemini's text response for the given audio + prompt.
    Replace internals with your Gemini SDK usage.
    """
    # Many Gemini setups prefer: transcribe audio first, then chat.
    # Placeholder:
    return "AI: (audio processed) I heard you—can you describe when these symptoms started?"

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
    photo_path: str  # e.g. "1/uuid.jpg" stored in media.photos (or sent directly by frontend)
    prompt: Optional[str] = None

class ProcessAudioRequest(BaseModel):
    convo_id: int
    recording_path: str  # e.g. "1/uuid.webm" stored in media.recordings
    prompt: Optional[str] = None

class ProcessResponse(BaseModel):
    ok: bool
    ai_text: str
    updated_message_row: Any

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Memento Python Backend")

@app.post("/process/image", response_model=ProcessResponse)
def process_image(req: ProcessImageRequest):
    # 1) Build public URL
    url = public_object_url("photos", req.photo_path)

    # 2) Fetch bytes
    img = fetch_bytes(url)

    # 3) Gemini
    prompt = req.prompt or "You are talking to an elderly user about their cherished item. Ask a warm, simple follow-up question."
    ai_text = gemini_analyze_image(img, prompt)

    # 4) Append to DB thread
    updated = append_ai_turn(
        convo_id=req.convo_id,
        ai_text=ai_text,
        kind="image",
        meta={"photo_url": url, "photo_path": req.photo_path},
    )

    # 5) Return for frontend to display
    return ProcessResponse(ok=True, ai_text=ai_text, updated_message_row=updated)

@app.post("/process/audio", response_model=ProcessResponse)
def process_audio(req: ProcessAudioRequest):
    # 1) Build public URL
    url = public_object_url("recordings", req.recording_path)

    # 2) Fetch bytes
    audio = fetch_bytes(url)

    # 3) Gemini
    prompt = req.prompt or "You are helping an elderly user describe what they said. Extract key details, then ask a short follow-up question."
    ai_text = gemini_analyze_audio(audio, prompt)

    # 4) Append to DB thread
    updated = append_ai_turn(
        convo_id=req.convo_id,
        ai_text=ai_text,
        kind="audio",
        meta={"recording_url": url, "recording_path": req.recording_path},
    )

    # 5) Return
    return ProcessResponse(ok=True, ai_text=ai_text, updated_message_row=updated)