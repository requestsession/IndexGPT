import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import streamlit as st
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from config import BASE_MODEL_ID, LORA_MODEL_PATH, RESEARCH_FIELD

class StopOnSignal(StoppingCriteria):
    def __init__(self, stop_event):
        self.stop_event = stop_event
    def __call__(self, input_ids, scores, **kwargs):
        if self.stop_event and self.stop_event.is_set():
            return True
        return False

@st.cache_resource
def get_model_and_tokenizer():
    """缓存加载模型，避免重复占用显存"""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, use_fast=True)

    # 显存优化加载
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )

    # 检查 LoRA 路径是否存在，如果不存在则只返回 base_model
    if LORA_MODEL_PATH.exists():
        try:
            model = PeftModel.from_pretrained(base_model, str(LORA_MODEL_PATH))
            print(f"Loaded LoRA weights from {LORA_MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load LoRA: {e}, using base model.")
            model = base_model
    else:
        model = base_model

    model.eval()
    return tokenizer, model

def run_inference_stream(
    query,
    context_items,
    history=None,
    max_tokens=2048,
    stop_event=None,
    history_rounds=3,
):
    """结合 RAG 上下文和历史记忆进行流式推理"""
    tokenizer, model = get_model_and_tokenizer()

    # 构造上下文文本
    context_str = "\n\n".join([f"--- Excerpt {i+1} ---\n{item['text']}" for i, item in enumerate(context_items)])

    system_content = f"You are IndexGPT, a professional and helpful research assistant specializing in {RESEARCH_FIELD} developed by requestsession.\n\n" \
                     f"Your core capabilities:\n" \
                     f"1. Analyze provided research excerpts and answer questions based on them.\n" \
                     f"2. Summarize academic methodologies and findings.\n" \
                     f"3. Discuss general concepts in {RESEARCH_FIELD} when explicit excerpts are missing.\n\n" \
                     f"Guidelines:\n" \
                     f"- If relevant excerpts are provided below, prioritize them and cite them as references.\n" \
                     f"- If the query is about your identity or general capabilities, respond naturally as IndexGPT.\n" \
                     f"- Maintain an academic yet accessible tone.\n\n" \
                     f"Current Context (Excerpts from papers):\n{context_str}"

    messages = [{"role": "system", "content": system_content}]

    # 添加历史记录 (限制最近 6 条，即 3 轮对话)
    if history:
        history_window = max(1, int(history_rounds)) * 2
        messages.extend(history[-history_window:])

    # 添加当前问题
    messages.append({"role": "user", "content": query})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    stopping_criteria = StoppingCriteriaList([StopOnSignal(stop_event)]) if stop_event else None

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return streamer

def run_inference(query, context_items, max_tokens=512):
    """结合 RAG 上下文进行推理（非流式，保留兼容性）"""
    tokenizer, model = get_model_and_tokenizer()

    # 构造上下文文本
    context_str = "\n\n".join([f"--- Excerpt {i+1} ---\n{item['text']}" for i, item in enumerate(context_items)])

    prompt = f"""You are IndexGPT, an expert researcher in {RESEARCH_FIELD} developed by requestsession.
Use the following excerpts from research papers to answer the question precisely.
If the information is not in the excerpts, explain that based on the provided documents.

Context:
{context_str}

Question: {query}
Answer:"""

    messages = [
        {"role": "system", "content": f"You are IndexGPT, a helpful and precise research assistant specializing in {RESEARCH_FIELD} developed by requestsession."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)
