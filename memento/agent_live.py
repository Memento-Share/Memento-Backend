from typing import List, Optional, Literal, Any, Dict, TypedDict
import asyncio
from google import genai
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import queue
load_dotenv()

SR = 24000
CH = 1
DTYPE = np.int16

pcm_q: queue.Queue[bytes] = queue.Queue()
_leftover = bytearray()  # <-- keep extra bytes between callbacks

def audio_callback(outdata, frames, time, status):
    global _leftover

    if status:
        # If you see underflow here, you're not buffering enough
        print("SD status:", status)

    need = frames * CH * np.dtype(DTYPE).itemsize

    # Start with whatever we had left from last callback
    buf = _leftover
    _leftover = bytearray()

    # Pull more until we have enough
    while len(buf) < need:
        try:
            buf += pcm_q.get_nowait()
        except queue.Empty:
            break

    # If still not enough, pad with silence
    if len(buf) < need:
        buf += b"\x00" * (need - len(buf))

    # IMPORTANT: keep any extra for the next callback (don't discard!)
    if len(buf) > need:
        _leftover = buf[need:]
        buf = buf[:need]

    samples = np.frombuffer(buf, dtype=DTYPE).reshape(frames, CH)

    # Debug: if this prints ~0 always, you are feeding silence / wrong format
    # rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    # print("RMS:", rms)

    outdata[:] = samples


client = genai.Client()

# --- Live API config ---
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "output_audio_transcription": {},
    "system_instruction": "You are a helpful and friendly AI assistant.",
}
async def live_chat():
    # Prebuffer target: 300ms of mono int16 PCM @24kHz
    prebuffer_ms = 300
    prebuffer_bytes = int(SR * (prebuffer_ms / 1000) * CH * 2)  # 2 bytes per int16
    buffered = 0

    stream = sd.OutputStream(
        samplerate=SR,
        channels=CH,
        dtype="int16",
        callback=audio_callback,
        blocksize=0,
    )
    try :
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            turns = [{"role" : "user", "parts" : [{"text" : "Hello, my name is Swapnil. Can you introduce yourself?"}]},
                    {"role" : "model", "parts" : [{"text" : "Hi Swapnil! I'm an AI assistant created by Google. I'm here to help you with any questions or tasks you have. What can I do for you today?"}]}]
            await session.send_client_content(turns=turns, turn_complete=False)
            message = "What is Kanye West's famous song about Paris?"
            await session.send_client_content(turns=[{"role" : "user", "parts" : [{"text" : message}]}], turn_complete=True)
            started = False
            async for response in session.receive():
                if (response.server_content and response.server_content.model_turn):
                    for part in response.server_content.model_turn.parts:
                        if part.inline_data and isinstance(part.inline_data.data, bytes) and len(part.inline_data.data) > 0:
                            pcm_q.put(part.inline_data.data)
                            buffered += len(part.inline_data.data)
                            if (not started) and buffered >= prebuffer_bytes:
                                print("starting stream (prebuffer reached)")
                                stream.start()
                                started = True

                    if response.server_content.output_transcription:
                        print("Transcript:", response.server_content.output_transcription.text)
    
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    asyncio.run(live_chat())
                    
                    








