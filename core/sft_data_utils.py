import random
import re
from pathlib import Path

from config import RAG_INDEX_DIR


def stable_dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def extract_focus_terms(text: str, max_terms: int = 8):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text or "")
    stop = {
        "write", "using", "include", "focus", "bullet", "points", "words",
        "explicitly", "implied", "state", "extract", "summary", "methodology",
        "context", "findings", "implications"
    }
    terms = []
    for tok in tokens:
        low = tok.lower()
        if low in stop:
            continue
        terms.append(low)
    return stable_dedupe(terms)[:max_terms]


def build_query_variants(instruction: str, format_hint: str, max_variants: int = 12):
    base = instruction.strip()
    hints = format_hint.strip()
    terms = extract_focus_terms(f"{instruction} {format_hint}", max_terms=10)

    variants = [
        base,
        f"{base} {hints}",
        f"{base} research paper excerpt",
        f"{base} technical details constraints assumptions",
    ]

    for term in terms:
        variants.append(f"{base} {term}")
        variants.append(f"{base} about {term}")

    return stable_dedupe([v.strip() for v in variants if v.strip()])[:max_variants]


def load_rag_texts_fallback(min_chars: int = 120):
    texts_path = RAG_INDEX_DIR / "texts.json"
    if not texts_path.exists():
        return []
    try:
        import json
        texts = json.loads(texts_path.read_text(encoding="utf-8"))
        return [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= min_chars]
    except Exception:
        return []


def build_candidate_pool(instruction, format_hint, rag_query_fn, topk_per_query=40, max_pool=1200):
    variants = build_query_variants(instruction, format_hint)
    pool = []
    seen_excerpt = set()

    for query in variants:
        hits = rag_query_fn(query, topk=topk_per_query)
        for h in hits:
            text = (h.get("text") or "").strip()
            if not text:
                continue
            if text in seen_excerpt:
                continue
            seen_excerpt.add(text)
            pool.append(h)
            if len(pool) >= max_pool:
                break
        if len(pool) >= max_pool:
            break

    return pool


def extend_pool_with_fallback(pool, target_size=1500, seed=42):
    if len(pool) >= target_size:
        return pool

    fallback_texts = load_rag_texts_fallback()
    if not fallback_texts:
        return pool

    randomizer = random.Random(seed)
    randomizer.shuffle(fallback_texts)
    seen = {((x.get("text") or "").strip()) for x in pool}

    for text in fallback_texts:
        if text in seen:
            continue
        pool.append({"text": text, "source": "fallback_pool", "page": 0, "score": 0.0})
        seen.add(text)
        if len(pool) >= target_size:
            break

    return pool
