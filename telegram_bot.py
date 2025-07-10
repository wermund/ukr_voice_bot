import os
import tempfile
import json
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, MessageHandler, CommandHandler,
    ContextTypes, filters
)

import whisper
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

# === Завантаження .env ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не знайдено в .env")
if not ELEVEN_KEY:
    raise RuntimeError("❌ ELEVENLABS_API_KEY не знайдено в .env")

# === Завантаження моделі Whisper ===
whisper_model = whisper.load_model("base")

# === Завантаження FAQ ===
FAQ_FILE = "faqs_ua.json"
if not os.path.exists(FAQ_FILE):
    raise FileNotFoundError("❌ Файл faqs_ua.json не знайдено")
with open(FAQ_FILE, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# === Підготовка ембеддингів ===
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
questions = [item["question"] for item in faq_data]
embeddings = embed_model.encode(questions, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# === Функції ===
def find_answer(query: str) -> str:
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vec), k=1)
    return faq_data[I[0][0]]["answer"]

def transcribe(file_path: str) -> str:
    result = whisper_model.transcribe(file_path, language="uk")
    return result["text"].strip()

def tts_elevenlabs(text: str) -> str:
    url = "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL/stream"
    headers = {"xi-api-key": ELEVEN_KEY}
    payload = {"text": text}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp.write(response.content)
        temp.flush()
        return temp.name
    else:
        print("TTS Error:", response.status_code, response.text)
        return None

# === Обробка голосу ===
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice_file = await update.message.voice.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_ogg:
        await voice_file.download_to_drive(temp_ogg.name)
        wav_path = temp_ogg.name.replace(".ogg", ".wav")
        AudioSegment.from_file(temp_ogg.name).export(wav_path, format="wav")

    query = transcribe(wav_path)
    if not query:
        await update.message.reply_text("⚠️ Не почув питання. Спробуй ще раз.")
        return

    await update.message.reply_text(f"🎧 Ви сказали: {query}")
    answer = find_answer(query)
    await update.message.reply_text(f"🤖 {answer}")

    voice_path = tts_elevenlabs(answer)
    if voice_path:
        with open(voice_path, "rb") as audio:
            await update.message.reply_voice(audio)

# === Обробка тексту ===
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return
    answer = find_answer(query)
    await update.message.reply_text(f"🤖 {answer}")

# === Стартова команда ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎤 Надішли голосове або текстове запитання українською!")

# === Запуск ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🚀 Бот запущено!")
    app.run_polling()

if __name__ == "__main__":
    main()