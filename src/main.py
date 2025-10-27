from dotenv import load_dotenv

load_dotenv()

import os
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler

from client.orchestrator import orchestrator

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
print(TELEGRAM_TOKEN)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""Assalamualaikum Akhi""")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # retrieve the user information
    user = update.message.from_user
    user_id = user.id

    user_message = update.message.text

    response = await orchestrator(user_id, user_message, model_name="gpt-4o-mini")

    await update.message.reply_text(response)

if __name__ == "__main__":
    print("initializing bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, chat))
    app.run_polling()
    print("chatbot successfully initialized")
