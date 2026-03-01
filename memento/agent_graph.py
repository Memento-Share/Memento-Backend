from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Optional, Literal, Any, Dict, TypedDict
import dotenv
import os
import base64
from pathlib import Path
from google.cloud import texttospeech
import google

dotenv.load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
TTS_LOCATION = os.getenv("GOOGLE_CLOUD_REGION")

API_ENDPOINT = (
    f"{TTS_LOCATION}-texttospeech.googleapis.com"
    if TTS_LOCATION != "global"
    else "texttospeech.googleapis.com"
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash"
TTS_MODEL = "gemini-2.5-flash-lite-preview-tts"

gemini = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=GEMINI_API_KEY)

class MementoState(MessagesState):
    request_stage : Literal["image", "audio"]
    voice : bytes | None

def init_photo_state(photo_bytes: bytes, mime_type: str) -> MementoState:
    sys_prompt : str = (
    "You are a warm, patient interviewer helping an elderly person tell stories. "
    "Given a photo of a keepsake/heirloom, ask ONE open-ended question to start."
    )
    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    messages = [SystemMessage(content=sys_prompt),
                HumanMessage(content = [{"type" : "image", "base64" : b64, "mime_type": mime_type}])]
    return MementoState(messages=messages, request_stage="image", voice=None)

def init_audio_state(audio_bytes: bytes, mime_type: str, messages: List) -> MementoState:
    sys_prompt : str = (
    "You are a warm, patient interviewer helping an elderly person tell stories. "
    "Given a short audio clip of the person's voice, ask ONE open-ended follow up question to continue the conversation."
    )
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    new_messages = messages + [SystemMessage(content=sys_prompt), HumanMessage(content = [{"type" : "audio", "base64" : b64, "mime_type": mime_type}])]
    return MementoState(messages=new_messages, request_stage="audio", voice=None)

def photo_node(state: MementoState) -> Dict[Literal["messages"], Any]:
    response = gemini.invoke(state["messages"])
    print("Photo node response:", response)
    return {"messages": [response]}

def audio_node(state: MementoState) -> Dict[Literal["messages"], Any]:
    response = gemini.invoke(state["messages"])
    print("AUDIO node response:", response)
    return {"messages": [response]}

def tts_node(state: MementoState) -> MementoState:
    client = texttospeech.TextToSpeechClient(client_options={"api_endpoint": API_ENDPOINT})
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    prompt = "You are a warm, patient interviewer helping an elderly person tell stories. Speak in a friendly, conversational tone."
    ai_response = state["messages"][-1].content
    synthesis_input = texttospeech.SynthesisInput(text=ai_response)
    response = client.synthesize_speech(input=synthesis_input, audio_config=audio_config, voice=texttospeech.VoiceSelectionParams(language_code="en-US", model_name = TTS_MODEL, name = "Achernar"))
    output_filepath = Path("test_media/response.mp3")
    with open(output_filepath, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to file: {output_filepath}")
    return MementoState(messages=state["messages"], request_stage=state["request_stage"], voice=response.audio_content)


def router(state): 
    if state["request_stage"] == "image":
        return "photo_node"
    elif state["request_stage"] == "audio":
        return "audio_node"


graph = StateGraph(MementoState)
graph.add_node("photo_node", photo_node)
graph.add_node("audio_node", audio_node)
graph.add_node("tts_node", tts_node)
graph.add_conditional_edges(START, router)
graph.add_edge("photo_node", "tts_node")
graph.add_edge("audio_node", "tts_node")
graph.add_edge("tts_node", END)
agent = graph.compile()

def run_agent(state : MementoState):
    result = agent.invoke(state)
    return result

def test_graph():
    test_audio = Path("test_media/monster.mp3").read_bytes()
    initial_state = init_audio_state(test_audio, "audio/mpeg", [])
    result = agent.invoke(initial_state)

if __name__ == "__main__":
    test_graph()