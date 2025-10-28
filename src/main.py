from dotenv import load_dotenv

load_dotenv()

import os, tempfile, requests
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler

from client.orchestrator import orchestrator
from servers.tools import image_to_data_uri

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
print(TELEGRAM_TOKEN)

# define the bot message when the user start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""Assalamualaikum Akhi""")

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

    image_data_uri = image_to_data_uri(file_bytes)
    user_question = update.message.caption or ""

    message = [
        {"type": "text", "text": user_question},
        {"type": "image_url", "image_url": image_data_uri},
    ]

    response = await orchestrator(user_id, message, model_name="gpt-4o-mini")
    await update.message.reply_text(response.content)    

# voice not chat handler
async def voice_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # retrieve the user information
    user = update.message.from_user
    user_id = user.id

    voice_note = update.message.voice
    file_id = voice_note.file_id

    file_object = await context.bot.get_file(file_id)

    # Create a temporary file path. Telegram voice notes are OGG/Opus.
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
        temp_path = temp_file.name
        
        # Download the file
        await file_object.download_to_drive(custom_path=temp_path)
    
    # send the message to speech to text model
    with open(temp_path, 'rb') as f:
        files = {'file': ('voice.ogg', f, 'audio/ogg')}
        response = requests.post(
                EXTERNAL_API_URL, 
                files=files, 
                # headers=headers,
                timeout=30 # Set a timeout for the request
            )
        string_message = response.text

    response = await orchestrator(user_id, string_message, model_name="gpt-4o-mini")

    await update.message.reply_text(response)

if __name__ == "__main__":
    print("initializing bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, chat))
    app.add_handler(MessageHandler(filters.VOICE, voice_chat))
    app.run_polling()
    print("chatbot successfully initialized")
