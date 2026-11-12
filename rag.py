import asyncio

import chromadb
from ollama import AsyncClient

from config import (CHAT_MODEL, COLLECTION, DB_PATH, EMBED_MODEL,
                    MAX_DISTANCE, NUM_CTX, TOP_K)

client = AsyncClient()
col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
sem = asyncio.Semaphore(2)

NOT_FOUND = (
    "Информация не найдена\n\n"
    "Свяжитесь с оператором"
)

SYSTEM = """You are an assistant that helps students with university admission.
Rules:
- Answer ONLY using the provided context.
- If the context does not contain the answer, say you don't know and suggest contacting the admissions office.
- Never invent deadlines, prices, requirements or contact details.
- Answer in the same language as the student's question.
- Be short and clear. Use a short list when giving several items."""


async def rewrite(question: str, history: list) -> str:
    """Turn a follow-up into a standalone question for retrieval."""
    if not history:
        return question
    convo = "\n".join(f"{role}: {text}" for role, text in history[-4:])
    prompt = (
        "Rewrite the student's last question so it is fully self-contained, "
        "using the conversation for context. Keep the same language. "
        "Output only the rewritten question.\n\n"
        f"Conversation:\n{convo}\n\nLast question: {question}\n\nStandalone question:"
    )
    resp = await client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 2048},
        think=False,
    )
    return resp["message"]["content"].strip() or question


async def answer(question: str, history: list):
    """Returns (answer_text, found_in_documents)."""
    async with sem:
        standalone = await rewrite(question, history)
        emb = (await client.embed(model=EMBED_MODEL, input=standalone))["embeddings"]
        res = await asyncio.to_thread(col.query, query_embeddings=emb, n_results=TOP_K)

        documents = (res.get("documents") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]

        hits = [
            (doc, meta)
            for doc, dist, meta in zip(documents, distances, metadatas)
            if dist <= MAX_DISTANCE
        ]
        if not hits:
            return NOT_FOUND, False

        context = "\n\n---\n\n".join(doc for doc, _ in hits)
        resp = await client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {standalone}"},
            ],
            options={"temperature": 0.2, "num_ctx": NUM_CTX},
            think=False,
        )
        return resp["message"]["content"].strip(), True