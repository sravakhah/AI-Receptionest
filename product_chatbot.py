import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
import pyttsx3
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Set up model and database (load once)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="product_db", embedding_function=embedding)
retriever = db.as_retriever()
llm = ChatOllama(model="phi")  # Change to "llama3" or "mistral" if desired

engine = pyttsx3.init()
engine.setProperty("rate", 160)

whisper_model = whisper.load_model("base")

def record_audio(duration=5, samplerate=44100):
    print("🎤 Recording for", duration, "seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def transcribe_audio(audio_data, samplerate=44100):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        import scipy.io.wavfile
        scipy.io.wavfile.write(tmpfile.name, samplerate, audio_data)
        filename = tmpfile.name

    result = whisper_model.transcribe(filename)
    os.remove(filename)
    return result["text"]

def get_ai_response():
    try:
        audio_data = record_audio()
        user_input = transcribe_audio(audio_data)
        print("📝 User said:", user_input)

        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"

        response = llm.invoke(prompt)
        #engine.say(response.content)  # or str(response) if .content doesn't exist
        assistant_text = getattr(response, "content", str(response))
        engine.say(assistant_text)
        engine.runAndWait()
        #return str(response)
        #print(type(response))
        #assistant_text = getattr(response, "content", str(response))
        #return str(response.context)
        return assistant_text

    except Exception as e:
        return f"⚠️ Error: {e}"
