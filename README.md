# 🇺🇦 Voice FAQ Bot

Два боти з голосовим інтерфейсом українською:
- `telegram_bot.py` — відповідає в Telegram [Протестувати: http://t.me/viknodeliverybot]
- `bot.py` — офлайн-бот, що слухає з мікрофона

---

## ⚙️ Встановлення

```bash
git clone https://github.com/wermund/ukr_voice_bot.git
cd ukr_voice_bot
pip install -r requirements.txt
```

---

## 🗝️ Налаштування

1. Створи `.env` на основі прикладу:

```bash
cp .env.example .env
```

2. Введи свої ключі в `.env`:

```env
TELEGRAM_BOT_TOKEN=your_telegram_token
ELEVENLABS_API_KEY=your_eleven_key
```

---

## 🚀 Запуск ботів

### ▶️ Telegram-бот

```bash
python telegram_bot.py
```

> Після запуску бот відповідає на голосові та текстові запити.
> Приклади питань, які можна задати: "Як зробити замовлення?", "Який термін доставки?", "Чи можна повернути товар?"

---

### 🎙️ Локальний офлайн-бот

```bash
python bot.py
```

> ⚠️ Потрібна модель `vosk-model-uk-v3`. Завантажити можна тут:  
> https://alphacephei.com/vosk/models

---

## 🧠 Формат файлу `faqs_ua.json`

```json
[
  {
    "question": "Як замовити товар?",
    "answer": "Щоб замовити товар, перейдіть на сайт і натисніть кнопку 'Купити'."
  },
  {
    "question": "Скільки триває доставка?",
    "answer": "Доставка зазвичай займає 1–3 дні по Україні."
  }
]
```

---

## 🛠️ Технології

- Python 3.10+
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Vosk](https://alphacephei.com/vosk/)
- [ElevenLabs TTS](https://elevenlabs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ukrainian TTS](https://github.com/robinhad/ukrainian-tts)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)

---

## 📄 Ліцензія

Проєкт поширюється за умовами ліцензії **MIT**.

---

## 👤 Автор

**@VikNo** | 2025  
Пиши, якщо маєш ідеї або хочеш доєднатися 💙💛

---
