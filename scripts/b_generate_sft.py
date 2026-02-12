import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import re

from a_rag import rag_query
from config import RESEARCH_FIELD, DATA_DIR, SFT_DATA_PATH
from core.llm_sft import call_llm
from core.sft_data_utils import build_candidate_pool, extend_pool_with_fallback

DATA_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    (
        "Summarize the methodology or core approach proposed in this excerpt.",
        "Write 3-6 bullet points. Mention specific techniques, models, or frameworks used.",
    ),
    (
        "State the research question or primary objective addressed in this excerpt.",
        "1-2 sentences. Include the specific problem or phenomenon being investigated.",
    ),
    (
        "Extract limitations, constraints, or boundary conditions explicitly stated or strongly implied.",
        "3-8 bullet points. Each <= 20 words. Focus on scope and potential biases.",
    ),
    (
        "Extract future directions, open questions, or broader implications mentioned or implied.",
        "3-8 bullet points. Use technical language relevant to the field.",
    ),
    (
        "Extract key results, metrics, empirical findings, or major conclusions in this excerpt.",
        "Bullet points. If no specific results are present, output exactly: NOT_FOUND.",
    ),
    (
        "Rewrite as a concise academic summary: Context -> Methodology -> Key Findings -> Implications.",
        "One paragraph (100-180 words). Provide a coherent synthesis of the excerpt's contribution.",
    ),
]

TOPK_EXCERPTS = 40
TARGET_PER_TASK = 200
MAX_CANDIDATE_POOL = 1800
MIN_ANSWER_CHARS = 60

BAD_PATTERNS = [
    r"^-\s*use\s+only\s+the\s+given\s+text",
    r"^-\s*do\s+not\s+add",
    r"^-\s*write\s+in\s+academic",
    r"^-\s*limit\s+to",
    r"^instructions?:",
    r"^task:",
]


def looks_like_instructions(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 40:
        return True
    lines = [ln.strip().lower() for ln in t.splitlines() if ln.strip()]
    hit = 0
    for ln in lines[:10]:
        for pat in BAD_PATTERNS:
            if re.search(pat, ln):
                hit += 1
                break
    return hit >= 2


def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def make_prompt(excerpt: str, instruction: str, format_hint: str) -> str:
    return f"""
You are IndexGPT, a senior researcher in temporal graph algorithms developed by requestsession.

Rules:
- Use ONLY the information in the excerpt.
- Preserve method names, symbols, and constraints.
- Do NOT output these rules or any meta-instructions.
- If evidence is insufficient, output exactly: NOT_FOUND.

Excerpt:
\"\"\"{excerpt}\"\"\"

Instruction:
{instruction}

Format requirements:
{format_hint}
""".strip()


def pack_text(instruction: str, excerpt: str, answer: str) -> str:
    return f"""### Instruction:
{instruction}

### Input:
{excerpt}

### Response:
{answer}
"""


def generate_task_samples(
    task_idx: int,
    instruction: str,
    format_hint: str,
    writer,
    seen_global: set,
) -> int:
    print(f"Processing Task {task_idx + 1}/{len(TASKS)}: {instruction}")
    collected = 0
    rejected_not_found = 0
    rejected_format = 0
    rejected_short = 0

    pool = build_candidate_pool(
        instruction=instruction,
        format_hint=format_hint,
        rag_query_fn=rag_query,
        topk_per_query=TOPK_EXCERPTS,
        max_pool=MAX_CANDIDATE_POOL,
    )
    pool = extend_pool_with_fallback(pool, target_size=MAX_CANDIDATE_POOL, seed=task_idx + 42)
    print(f"  Candidate pool size: {len(pool)}")

    for h in pool:
        if collected >= TARGET_PER_TASK:
            break

        excerpt = (h.get("text") or "").strip()
        if not excerpt:
            continue

        key = stable_hash(excerpt + "||" + instruction)
        if key in seen_global:
            continue
        seen_global.add(key)

        prompt = make_prompt(excerpt, instruction, format_hint)
        answer = call_llm(prompt).strip()

        if answer == "NOT_FOUND":
            rejected_not_found += 1
            continue
        if looks_like_instructions(answer):
            rejected_format += 1
            continue
        if len(answer) < MIN_ANSWER_CHARS:
            rejected_short += 1
            continue

        writer.write(json.dumps({"text": pack_text(instruction, excerpt, answer)}, ensure_ascii=False) + "\n")
        collected += 1

        if collected % 10 == 0:
            print(
                f"  [Progress] Task {task_idx + 1}/{len(TASKS)}: Collected "
                f"{collected}/{TARGET_PER_TASK} samples..."
            )

    if collected < TARGET_PER_TASK:
        print(
            f"  [Warning] Task {task_idx + 1} pool exhausted before target "
            f"({collected}/{TARGET_PER_TASK}). Rejected: "
            f"NOT_FOUND={rejected_not_found}, FORMAT={rejected_format}, SHORT={rejected_short}."
        )

    print(f"[Done] Task {task_idx + 1} complete.\n")
    return collected


def main():
    out_path = Path(SFT_DATA_PATH)
    seen = set()

    print("Starting SFT data generation...")
    print(f"Total tasks: {len(TASKS)} | Target per task: {TARGET_PER_TASK}\\n")

    total_collected = 0
    with out_path.open("w", encoding="utf-8") as f:
        for task_idx, (instruction, format_hint) in enumerate(TASKS):
            total_collected += generate_task_samples(
                task_idx,
                instruction,
                format_hint,
                f,
                seen,
            )

        print(f"Appending identity samples for IndexGPT ({RESEARCH_FIELD})...")
        identity_samples = [
            ("Who are you?", f"I am IndexGPT, a senior researcher specializing in {RESEARCH_FIELD} developed by requestsession."),
            ("What is your name?", "My name is IndexGPT."),
            ("Who developed you?", "I was developed by requestsession."),
            ("你是谁？", f"我是 IndexGPT，由 requestsession 开发的 {RESEARCH_FIELD} 领域专业研究员。"),
            ("你叫什么名字？", "我叫 IndexGPT。"),
            ("谁开发了你？", "我由 requestsession 开发。"),
        ]
        for q, a in identity_samples:
            f.write(json.dumps({"text": f"### Instruction:\n{q}\n\n### Response:\n{a}\n"}, ensure_ascii=False) + "\n")

    lines = sum(1 for _ in out_path.open("r", encoding="utf-8"))
    print(f"Done. Collected {total_collected} task samples. Wrote {out_path} with ~{lines} lines.")


if __name__ == "__main__":
    main()
