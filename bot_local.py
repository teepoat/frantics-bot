import time
from typing import Final
import re
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from typing import Optional
import random
import os
from dotenv import load_dotenv
from models.seq2seq.model import Seq2SeqChatbot
import torch

load_dotenv()

TOKEN: Final = os.environ.get("TOKEN")
BOT_USERNAME: Final = os.environ.get("BOT_USERNAME")
CHAT_ID: Final = int(os.environ.get("CHAT_ID"))

CHECKPOINT_PATH: Final = "models/seq2seq/checkpoint/150_checkpoint.tar"

ROMANTIKI_GIF_ID: Final = "CgACAgIAAxkBAAE4zMlojLmMwqrxG5e2rnYS2f9_PZZgVwACL2oAAjbWyUqiyR5II6u6YDYE"
BEZUMTSI_GIF_ID: Final = "CgACAgIAAxkBAAE4zMlojLmMwqrxG5e2rnYS2f9_PZZgVwACL2oAAjbWyUqiyR5II6u6YDYE"

last_gif_sent = 1.0
gif_sent_cooldown = 180.0
response_chance =  1.0

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chatbot = Seq2SeqChatbot(500, 8856, 2, 2, 0.1, device)
chatbot.load_checkpoint(CHECKPOINT_PATH)
chatbot.eval_mode()


def handle_response(text: str) -> Optional[str]:
    if random.random() < response_chance:
        return chatbot(text)
    return None


def edit_response(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = re.sub(r'\s+([,.!?;])\s+', r'\1 ', text)
    return text


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.chat_id == CHAT_ID:
        global last_gif_sent
        if "роман" in update.message.text.lower() and \
                time.time() - last_gif_sent >= gif_sent_cooldown:
            await context.bot.send_animation( chat_id=update.message.chat_id, animation=ROMANTIKI_GIF_ID)
            last_gif_sent = time.time()
        elif "безу" in update.message.text.lower() and \
                time.time() - last_gif_sent >= gif_sent_cooldown:
            await context.bot.send_animation(chat_id=update.message.chat_id, animation=BEZUMTSI_GIF_ID)
            last_gif_sent = time.time()
        else:
            text = update.message.text.replace(BOT_USERNAME, '').strip().lower()
            response = edit_response(handle_response(text))
            if response:
                await context.bot.sendMessage(update.message.chat_id, response, reply_to_message_id=update.message.id)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.message.from_user.username} in {update.message.chat.type} "
          f"chat caused error \"{context.error}\"\n"
          f"{update}\"")


def main() -> None:
    """Run the bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT, handle_message))
    application.add_error_handler(error)

    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == '__main__':
    print("Running main...")
    main()
