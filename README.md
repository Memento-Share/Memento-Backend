# Memento – Backend

The Memento Backend powers the intelligence layer of the Memento app. It handles multimodal AI processing (image + audio), maintains conversation state, generates voice responses, and securely stores memory artifacts for long-term preservation. It also hosts endpoints and databases on Supabase supporting all the core functionalities of the product.

This repository contains the **FastAPI-based backend service and langgraph class** for Memento.

---

# Purpose

The backend is responsible for:

- Processing user-submitted images and audio recordings
- Calling Gemini multimodal model and handles its integration with the frontend
- Maintaining structured conversation state and messages thread
- Generating AI voice responses (TTS)
- Persisting memory threads, members with access and media libraries in Supabase
- Protecting API keys and enforcing security, making sure that the frontend is abstracted from Gemini

---

# Tech Stack

## Core Framework
- **FastAPI**
- **Python**
- **Pydantic**
- **Langchain**
- **Supabase**

## AI Layer
- **Gemini Developer API**
  - Multimodal reasoning (image + audio)
  - Audio understanding
  - Text + audio generation

## State & Orchestration
- **LangGraph**
- Structured conversation state
- Conditional stage transitions 

## Storage
- **Supabase**
  - Storage (images + audio files)
  - Postgres (message threads)
  - RPC functions (`append_turns`)

---

# Architecture Overview

Frontend -> FastAPI Backend ->  Gemini Multimodal Model -> Supabase Storage + Database  -> Frontend

Image flow:  

1. Frontend uploads image to Supabase.
2. Backend receives image URL.
3. Backend fetches image bytes.
4. Gemini is called with image + initial prompt.
5. AI asks first guided question

Audio flow:

1. Frontend uploads recorded audio.
2. Backend fetches audio bytes.
3. Gemini processes:
   - Audio understanding
   - Contextual reasoning
4. Gemini generates:
   - Text reply
   - Audio reply (TTS)
5. Response is stored in Supabase thread.

# LangGraph State

We extend a MessagesState:

```python
class MementoState(MessagesState):
    request_stage: Literal["image", "audio"]

The graph routes based on request_stage, appends new messages and persists state after execution.
 
Preserving stories and legacies! 
