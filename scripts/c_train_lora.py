import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from config import BASE_MODEL_ID, LORA_MODEL_PATH, SFT_DATA_PATH

MODEL_ID = os.environ.get("MODEL_ID", BASE_MODEL_ID)
DATA_PATH = os.environ.get("DATA_PATH", str(SFT_DATA_PATH))
OUT_DIR = os.environ.get("OUT_DIR", str(LORA_MODEL_PATH))

os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)

ds = load_dataset("json", data_files=DATA_PATH, split="train")


def format_example(ex):
    return f"""### Instruction
{ex["instruction"]}

### Input
{ex["input"]}

### Response
{ex["output"]}"""


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

training_args = SFTConfig(
    output_dir=str(Path(OUT_DIR).parent / "out_sft"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    max_length=1024,
    push_to_hub=False,
    report_to=[],
    logging_strategy="steps",
    logging_steps=10,
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved LoRA to:", OUT_DIR)
