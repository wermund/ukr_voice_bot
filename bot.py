import os
import queue
import json
import sounddevice as sd
import vosk
from ukrainian_tts.tts import TTS, Voices, Stress
import simpleaudio as sa
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Налаштування ===
SAMPLE_RATE = 16000
MODEL_PATH = "vosk-model-uk-v3"
RESPONSES_FILE = "faqs_ua.json"

# === Перевірка моделей ===
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Модель Vosk не знайдено у {MODEL_PATH}")

if not os.path.exists(RESPONSES_FILE):
    raise RuntimeError(f"❌ Файл {RESPONSES_FILE} не знайдено.")

# === Завантаження Vosk і FAQ
model = vosk.Model(MODEL_PATH)

with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# === Ініціалізація трансформера
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
questions = [pair["question"] for pair in faq_data]
embeddings = embed_model.encode(questions, normalize_embeddings=True)
index = faiss.IndexFlatIP(len(embeddings[0]))
index.add(np.array(embeddings))

# === Черга аудіо
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(f"⚠️ {status}")
    q.put(bytes(indata))

def find_answer(query):
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vec), k=1)
    return faq_data[I[0][0]]["answer"]

def speak(text, output_path="output.wav"):
    tts = TTS()
    wav_bytes_io, _ = tts.tts(
        text,
        voice=Voices.Dmytro.value,
        stress=Stress.Dictionary.value
    )
    with open(output_path, "wb") as f:
        f.write(wav_bytes_io.getbuffer())

    wave_obj = sa.WaveObject.from_wave_file(output_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def listen_and_respond():
    print("🎤 Слухаю запит українською...")
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result_json = recognizer.Result()
                result = json.loads(result_json)
                text = result.get("text", "")
                if text:
                    print(f"👂 Ви сказали: {text}")
                    answer = find_answer(text)
                    print(f"🤖 Відповідь: {answer}")
                    speak(answer)
                    break

if __name__ == "__main__":
    listen_and_respond()