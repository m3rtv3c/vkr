import logging
import time
from collections import defaultdict

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from config import BOT_TOKEN, COOLDOWN_SEC, MAX_HISTORY
from rag import answer

logging.basicConfig(level=logging.INFO)

histories = defaultdict(list)   # user_id -> [(role, text), ...]
last_request = {}               # user_id -> timestamp

WELCOME = (
    "Привет! Я могу ответить на вопросы о поступлении: сроки подачи документов, необходимые документы, "
    "образовательные программы, стоимость обучения, стипендии и многое другое.\n\n"
    "Просто напишите свой вопрос. Используйте /clear, чтобы начать новый диалог.\n\n"
    "Важные даты всегда уточняйте в приёмной комиссии."
)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME)


async def clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    histories.pop(update.effective_user.id, None)
    await update.message.reply_text("Conversation cleared.")


async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    now = time.time()
    if now - last_request.get(user_id, 0) < COOLDOWN_SEC:
        await update.message.reply_text("Please wait a few seconds before the next question.")
        return
    last_request[user_id] = now

    if len(text) > 1000:
        await update.message.reply_text("Your message is too long. Please shorten it.")
        return

    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        reply, found = await answer(text, histories[user_id])
    except Exception:
        logging.exception("RAG error")
        await update.message.reply_text("Sorry, something went wrong. Please try again later.")
        return

    if not found:
        with open("unanswered.log", "a", encoding="utf-8") as f:
            f.write(text.replace("\n", " ") + "\n")

    histories[user_id] += [("Student", text), ("Assistant", reply)]
    histories[user_id] = histories[user_id][-MAX_HISTORY:]

    for i in range(0, len(reply), 4000):   # Telegram limit is 4096 characters
        await update.message.reply_text(reply[i:i + 4000])


app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("help", start))
app.add_handler(CommandHandler("clear", clear))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
app.run_polling()