#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
#from langchain_community.chat_models import ChatOllama

# ✅ New
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from langchain.chains import RetrievalQA

import whisper                      # For speech recognition
import pyttsx3                     # For text-to-speech
import sounddevice as sd          # To record from your microphone
import numpy as np
import wave
import tempfile
import os

# --- Load Embeddings & Product Knowledge ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="product_db", embedding_function=embedding)

# --- Load Local Language Model (like mistral or phi) ---
llm = ChatOllama(model="phi")  # Change to "mistral" or "llama3" if you like

# --- Retrieval-Based QA Chain ---
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# --- Setup Whisper Model for Voice Recognition ---
whisper_model = whisper.load_model("base")  # Use "tiny" for faster but less accurate results

# --- Setup TTS Engine ---
engine = pyttsx3.init()

def record_voice(filename="voice_input.wav", duration=5, fs=44100):
    print("🎤 Listening... (Speak now)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(audio.tobytes())
    return filename

# --- Chat Loop ---
print("🗣️ Say something or type 'q' to quit.")
while True:
    filename = tempfile.mktemp(suffix=".wav")
    record_voice(filename)

    result = whisper_model.transcribe(filename)
    user_query = result["text"].strip()
    print(f"🧑 You said: {user_query}")

    if user_query.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    response = qa.run(user_query)
    print("🤖", response)

    engine.say(response)
    engine.runAndWait()

    os.remove(filename)
