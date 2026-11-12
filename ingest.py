import pathlib
import re

import chromadb
import ollama

from config import COLLECTION, DB_PATH, EMBED_MODEL


def read_file(path: pathlib.Path):
    if path.suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if path.suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return None

def excel_chunks(path: pathlib.Path, rows_per_chunk: int = 5):
    import pandas as pd

    if path.suffix == ".csv":
        sheets = {"data": pd.read_csv(path, dtype=str)}
    elif path.suffix == ".xls":
        sheets = pd.read_excel(
            path,
            sheet_name=None,
            dtype=str,
            engine="xlrd"
        )
    elif path.suffix == ".xlsx":
        sheets = pd.read_excel(
            path,
            sheet_name=None,
            dtype=str,
            engine="openpyxl"
        )
    else:
        return []

    out = []

    for sheet, df in sheets.items():
        df = df.dropna(how="all").fillna("")

        lines = []

        for _, row in df.iterrows():
            parts = [
                f"{col}: {val}"
                for col, val in row.items()
                if str(val).strip()
            ]

            if parts:
                lines.append("; ".join(parts))

        for i in range(0, len(lines), rows_per_chunk):
            block = "\n".join(lines[i:i + rows_per_chunk])
            out.append(f"# {path.stem} / {sheet}\n{block}")

    return out

def chunk(text: str, size: int = 1000, overlap: int = 150):
    sections = [s.strip() for s in re.split(r"(?m)^(?=#{1,3} )", text) if s.strip()]
    out = []
    for section in sections:
        heading = section.splitlines()[0] if section.startswith("#") else ""
        if len(section) <= size:
            out.append(section)
            continue
        i = 0
        while i < len(section):
            piece = section[i:i + size]
            if i > 0 and heading:
                piece = heading + "\n" + piece
            out.append(piece)
            i += size - overlap
    return out


client = chromadb.PersistentClient(path=DB_PATH)
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass
col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

ids, docs, metas = [], [], []
for path in sorted(pathlib.Path("data").glob("**/*")):
    if not path.is_file():
        continue
    if path.suffix in {".xlsx", ".csv", ".xls"}:
        pieces = excel_chunks(path)
    else:
        text = read_file(path)
        if not text:
            continue
        pieces = chunk(text)
    for n, c in enumerate(pieces):
        ids.append(f"{path.name}-{n}")
        docs.append(c)
        metas.append({"source": path.name})

BATCH = 32
for i in range(0, len(docs), BATCH):
    batch = docs[i:i + BATCH]
    embs = ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"]
    col.add(
        ids=ids[i:i + BATCH],
        documents=batch,
        embeddings=embs,
        metadatas=metas[i:i + BATCH],
    )

print(f"Indexed {len(docs)} chunks from {len(set(m['source'] for m in metas))} files")