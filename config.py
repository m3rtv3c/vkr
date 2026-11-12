import os

BOT_TOKEN = os.environ["BOT_TOKEN"]

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen3:8b"

DB_PATH = "db"
COLLECTION = "admission"

TOP_K = 5            # chunks passed to the LLM
MAX_DISTANCE = 0.6   # cosine distance; larger = less similar. Tune this.
NUM_CTX = 4096       # context window for the chat model
MAX_HISTORY = 6      # messages remembered per user
COOLDOWN_SEC = 3     # minimum seconds between questions per user