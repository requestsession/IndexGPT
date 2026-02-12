# IndexGPT

[中文说明](./README.zh-CN.md)

IndexGPT is a local research assistant workflow built around **RAG + SFT + LoRA**.
It supports paper ingestion, retrieval-based QA, SFT data generation, LoRA training, and web chat in one project.

## 0. Project Identity

- Repository: `https://github.com/requestsession/IndexGPT`
- Maintainer: `requestsession`

## 1. Environment

- Python: 3.10+
- Node.js: 18+
- OS: Windows/Linux (current repo is Windows-friendly)
- GPU: recommended for inference/training

Install dependencies:

```bash
pip install -r requirements.txt
cd web/frontend
npm install
```

## 2. Model Placement (Important)

This project uses **local model folders**. A model is not a single file, but a directory containing files such as:

- `config.json`
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`, ...)
- weights (`*.safetensors` / `pytorch_model*.bin`)

### Recommended placement

Put model folders under:

```text
workspace/rag_index/models/
```

Example:

```text
workspace/rag_index/models/
  Qwen2.5-7B-Instruct/
  bge-base-en-v1.5/
```

Then open Settings page and select:

- `Base LLM`
- `Embedding Model`

The backend scans `workspace/rag_index/models/` and lists available folders.

## 3. Start the App

From project root:

```bash
python web/backend/main.py
```

In another terminal:

```bash
cd web/frontend
npm run dev -- --host
```

Default ports:

- Backend: `8176` (from `config.py` / settings)
- Frontend: Vite default `5173`

## 4. UI Preview

### AI Assistant

![AI Assistant](./web/showcase/figs/1_AI_Assistant.png)

### Documents

![Documents](./web/showcase/figs/2_Documents.png)

### Training Lab

![Training Lab](./web/showcase/figs/3_Training_Lab.png)

### System Settings

![System Settings](./web/showcase/figs/4_System_Settings.png)

## 5. End-to-End Workflow

### Step A: Upload papers

- Open Documents tab
- Upload PDF files
- Files are stored in `workspace/papers/`

### Step B: Build RAG index

- Open Training Lab
- Click **RAG Indexing -> Execute**

Generated artifacts:

- `workspace/rag_index/parse/*.json`
- `workspace/rag_index/faiss.index`
- `workspace/rag_index/texts.json`
- `workspace/rag_index/meta.json`

### Step C: Generate SFT data

- Click **SFT Generation -> Execute**
- Output file: `workspace/data/sft.jsonl`

### Step D: Train LoRA

- Click **LoRA Training -> Execute**
- Output adapter path: `workspace/outputs/lora/`

### Step E: Chat / QA

- Open AI Assistant tab
- Ask questions against indexed papers
- Optional modes:
  - Compare mode (cross-paper comparison)
  - Web mode (web snippets)

## 6. Equivalent CLI Commands

Run from project root:

```bash
python -u scripts/a_rag.py
python -u scripts/b_generate_sft.py
python -u scripts/c_train_lora.py
```

Inference module entry:

```bash
python scripts/d_infer_lora.py
```

## 7. Key Paths

- `workspace/papers/`: uploaded PDFs
- `workspace/rag_index/`: retrieval index and metadata
- `workspace/data/sft.jsonl`: SFT dataset
- `workspace/outputs/lora/`: trained LoRA output
- `workspace/logs/`: runtime logs
- `workspace/chats/`: persisted chat sessions

## 8. Runtime Notes

- Heavy tasks are mutually exclusive (indexing/SFT/training cannot run together)
- Chat is blocked while a heavy task is running
- Sliding-window chat history is controlled by settings (`CHAT_HISTORY_ROUNDS`)
- If runtime directories under `workspace/` are deleted, they are auto-created again when `config.py` is loaded

## 9. Troubleshooting

### No model appears in Settings dropdown

Check:

1. Model directories exist under `workspace/rag_index/models/`
2. Each model folder is complete (not only one weight shard)
3. Refresh Settings page / restart backend

### Chat returns no sources

Usually means retrieval confidence is low. Re-check:

1. PDFs are readable text (not scanned images only)
2. Index has been rebuilt after adding PDFs
3. Query wording is specific enough

### Training fails quickly

Check:

1. `Base LLM` path is valid
2. `workspace/data/sft.jsonl` exists
3. GPU/VRAM availability and dependency versions

## 10. Citation

If this project helps your work, please cite:

```bibtex
@software{indexgpt,
  title = {IndexGPT},
  author = {requestsession},
  year = {2026},
  url = {https://github.com/requestsession/IndexGPT}
}
```

## 11. License

MIT

