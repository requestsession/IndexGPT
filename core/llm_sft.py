# llm_sft.py (Supports Local & OpenAI-compatible API)
import os
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import API_KEY, API_BASE_URL, API_MODEL_ID, SFT_MODEL_SOURCE, BASE_MODEL_ID

# --- API Client Cache ---
_API_CLIENT = None

def _get_api_client():
    """Lazy load the OpenAI client only when needed."""
    global _API_CLIENT
    if _API_CLIENT is None:
        _API_CLIENT = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )
    return _API_CLIENT

# --- Local Model Cache ---
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None

def _get_local_model():
    """Lazy load the local model and tokenizer for SFT generation."""
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is None:
        print(f"Loading local base model for SFT generation: {BASE_MODEL_ID}")
        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, use_fast=True)
        _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        _LOCAL_MODEL.eval()
    return _LOCAL_MODEL, _LOCAL_TOKENIZER

def call_llm(prompt: str,
             max_new_tokens: int = 512,
             temperature: float = 0.2,
             model: str = API_MODEL_ID) -> str:

    if SFT_MODEL_SOURCE == "api":
        # API Mode (Remote)
        api_client = _get_api_client()
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful researcher. Follow instructions strictly. Do not add unsupported claims."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()

    else:
        # Local Mode
        model_obj, tokenizer = _get_local_model()

        messages = [
            {"role": "system", "content": "You are a careful researcher. Follow instructions strictly. Do not add unsupported claims."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model_obj.device)

        with torch.no_grad():
            output_ids = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract new tokens
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
