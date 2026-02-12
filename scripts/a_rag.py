import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import re

import faiss
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

from config import BGE_MODEL_PATH, CHUNK_MAX_WORDS, PAPERS_DIR, PARSE_DIR, RAG_INDEX_DIR

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        if not BGE_MODEL_PATH.exists():
            _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        else:
            _model = SentenceTransformer(str(BGE_MODEL_PATH))
    return _model


def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages, max_words=CHUNK_MAX_WORDS):
    chunks = []
    cur = []
    cur_words = 0
    for p in pages:
        for para in p["text"].split("\n\n"):
            w = len(para.split())
            if w < 10:
                continue
            if cur_words + w > max_words:
                chunks.append({"text": " ".join(cur), "page": p["page"]})
                cur = []
                cur_words = 0
            cur.append(para)
            cur_words += w
    if cur:
        chunks.append({"text": " ".join(cur), "page": pages[-1]["page"]})
    return chunks


def is_bad_chunk(t: str) -> bool:
    low = t.lower()
    if "references" in low or "bibliography" in low:
        if re.search(r"\[\d+\]|doi:|arxiv:|pp\.", low):
            return True
    if "acknowledg" in low or "publisher's note" in low:
        return True
    return False


def run_rag_indexing():
    for pdf in PAPERS_DIR.glob("*.pdf"):
        pages = pdf_to_text(pdf)
        with open(PARSE_DIR / (pdf.stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

    all_texts = []
    all_metas = []
    model = get_embedding_model()

    for jf in PARSE_DIR.glob("*.json"):
        with open(jf, "r", encoding="utf-8") as fh:
            pages = json.load(fh)
        chunks = chunk_pages(pages)
        for i, c in enumerate(chunks):
            if is_bad_chunk(c["text"]):
                continue
            all_texts.append(c["text"])
            all_metas.append({
                "source": jf.stem + ".pdf",
                "page": c["page"],
                "chunk_id": i,
            })

    if not all_texts:
        return 0

    emb = model.encode(all_texts, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(emb))

    faiss.write_index(index, str(RAG_INDEX_DIR / "faiss.index"))
    with open(RAG_INDEX_DIR / "texts.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False)
    with open(RAG_INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(all_metas, f, ensure_ascii=False)

    return len(list(PAPERS_DIR.glob("*.pdf")))


def rag_query(q, topk=3):
    model = get_embedding_model()
    index_path = RAG_INDEX_DIR / "faiss.index"
    if not index_path.exists():
        return []

    index = faiss.read_index(str(index_path))
    with open(RAG_INDEX_DIR / "texts.json", "r", encoding="utf-8") as fh:
        texts = json.load(fh)
    with open(RAG_INDEX_DIR / "meta.json", "r", encoding="utf-8") as fh:
        metas = json.load(fh)

    q_emb = model.encode([q], normalize_embeddings=True)
    scores, idxes = index.search(np.array(q_emb), topk)

    results = []
    for i, idx in enumerate(idxes[0]):
        if 0 <= idx < len(texts) and idx < len(metas):
            results.append(
                {
                    "text": texts[idx],
                    "source": metas[idx].get("source", "Unknown"),
                    "page": metas[idx].get("page", 0),
                    "score": float(scores[0][i]),
                }
            )
    return results


if __name__ == "__main__":
    run_rag_indexing()
    print("Indexing done.")
