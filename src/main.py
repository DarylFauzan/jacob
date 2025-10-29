from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

import os, tempfile, requests
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler

from client.orchestrator import orchestrator
from servers.tools import image_to_base64

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# define the bot message when the user start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""Hai, aku Jacob! Asisten dari DOKU. Sekarang DOKU sudah menangani lebih dari 500,000,000 transaksi! Kamu mau kenal DOKU lebih dekat? Atau langsung join aja! Aku bakal bantu kamu di setiap langkahnya.""")

# text chat handler
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # retrieve the user information
    user = update.message.from_user
    user_id = user.id

    user_message = update.message.text

    response = await orchestrator(user_id, user_message, model_name="gpt-4o-mini")

    await update.message.reply_text(response)

# image handler
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_id = user.id

    photo = update.message.photo[-1]  # largest resolution
    file = await photo.get_file()
    file_bytes = requests.get(file.file_path).content

    image_data_base64 = image_to_base64(file_bytes)
    user_question = update.message.caption or ""

    message = [
        {"type": "text", "text": user_question},
        {
            "type": "image",
            "source_type": "base64",
            "data": image_data_base64,
            "mime_type": "image/jpeg",
        },
    ]

    try:
        response = await orchestrator(user_id, message, model_name="gpt-4o-mini")
        await update.message.reply_text(response)
    except Exception as e:  
        print(str(e))  
        await update.message.reply_text("Maaf untuk saat ini kami belum support fitur gambar. tolong ketik pesan Anda")

# voice not chat handler
async def voice_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # retrieve the user information
    user = update.message.from_user
    user_id = user.id

    voice_note = update.message.voice
    file_id = voice_note.file_id

    file_object = await context.bot.get_file(file_id)

    try:
        # Create a temporary file path. Telegram voice notes are OGG/Opus.
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            temp_ogg_path = temp_file.name
            
            # Download the file
            await file_object.download_to_drive(custom_path=temp_ogg_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            
            # Load the OGG file using pydub
            audio = AudioSegment.from_ogg(temp_ogg_path)
            
            # Export it to WAV format
            audio.export(temp_wav_path, format="wav")
            print(f"File converted from OGG to WAV: {temp_wav_path}")
        
        # send the message to speech to text model
        with open(temp_wav_path, 'rb') as f:
            files = {'file': ('voice.wav', f, 'audio/wav')}
            response = requests.post(
                    "https://jocob-voice-assistant-api-production.up.railway.app/api/speech/transcribe", 
                    files=files, 
                )
            
            if response.status_code != 200:
                raise ValueError(f"status_code: {response.status_code}, message: {response.text}")

            string_message = response.text
            print(string_message)

        response = await orchestrator(user_id, string_message, model_name="gpt-4o-mini")
        await update.message.reply_text(response)

    except Exception as e:
        print(str(e))
        await update.message.reply_text("Maaf untuk saat ini kami belum support fitur voice note. tolong ketik pesan Anda")


if __name__ == "__main__":
    print("initializing bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, chat))
    app.add_handler(MessageHandler(filters.VOICE, voice_chat))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
    print("chatbot successfully initialized")
