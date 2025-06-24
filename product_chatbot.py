import os
import subprocess
import tempfile
import uuid
import numpy as np
import sounddevice as sd
import pyttsx3
import whisper
from scipy.io.wavfile import write
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
AVATAR_IMAGE = "avatar.jpg"  # Make sure this image exists
SADTALKER_DIR = r"C:\Users\Asus\OneDrive\Desktop\Open source AI model\SadTalker"  # Adjust if needed
RESULT_DIR = os.path.join(SADTALKER_DIR, "results")

# Initialize models
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="product_db", embedding_function=embedding)
retriever = db.as_retriever()
llm = ChatOllama(model="phi")
whisper_model = whisper.load_model("base")
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def record_audio(duration=5, samplerate=44100):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def transcribe_audio(audio, samplerate=44100):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, samplerate, audio)
        path = tmpfile.name
    result = whisper_model.transcribe(path)
    os.remove(path)
    return result["text"]

def generate_talking_video(audio_path):
    subprocess.run([
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", AVATAR_IMAGE,
        "--result_dir", RESULT_DIR,
        "--enhancer", "gfpgan",
        "--preprocess", "full",
        "--still"
    ], cwd=SADTALKER_DIR)

def get_ai_response():
    try:
        audio = record_audio()
        user_input = transcribe_audio(audio)

        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join(doc.page_content for doc in docs)
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        response = llm.invoke(prompt)
        reply = getattr(response, "content", str(response))

        # Save voice to audio
        audio_out = os.path.join(tempfile.gettempdir(), f"reply_{uuid.uuid4().hex[:6]}.wav")
        engine.save_to_file(reply, audio_out)
        engine.runAndWait()

        # Generate talking avatar video
        generate_talking_video(audio_out)

        # Launch the video
        video_path = os.path.join(RESULT_DIR, "output.mp4")
        if os.path.exists(video_path):
            subprocess.Popen(["start", video_path], shell=True)

        return reply

    except Exception as e:
        return f"⚠️ Error: {e}"
