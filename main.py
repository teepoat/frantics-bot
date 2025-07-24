from typing import Final
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

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chatbot = Seq2SeqChatbot(500, 2, 2, 0.1, device)
chatbot.load_checkpoint(CHECKPOINT_PATH)
chatbot.eval_mode()

def handle_response(text: str) -> Optional[str]:
    response_chance = 1.0
    if random.random() < response_chance:
        return chatbot(text)
    return None


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.chat_id == CHAT_ID:
        text: str = update.message.text.replace(BOT_USERNAME, '').strip().lower()
        response: Optional[str] = handle_response(text)
        if response:
            await context.bot.sendMessage(update.message.chat_id, response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.message.from_user.username} in {update.message.chat.type} "
          f"chat caused error \"{context.error}\"\n"
          f"{update}\"")

def main() -> None:
    """Run the bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT, handle_message))
    application.add_error_handler(error)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    print("Running main...")
    # print(chatbot("test"))
    main()