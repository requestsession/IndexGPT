import sys
import os
from pathlib import Path

# Ensure project root is importable when launching from web/backend.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import shutil
import asyncio
import threading
import json
import subprocess
import signal
import uuid
import re
from datetime import datetime
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional
from dataclasses import asdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import PAPERS_DIR, TRAIN_LOG_PATH, CHATS_DIR, RAG_INDEX_DIR, BACKEND_PORT, SETTINGS_FILE, load_settings, BASE_MODEL_ID, BASE_DIR
from scripts.a_rag import run_rag_indexing, rag_query
from scripts.d_infer_lora import run_inference, run_inference_stream
from core.compare_utils import select_compare_sources, build_compare_query
from openai import OpenAI
from core.resource_monitor import ResourceMonitor
from core.chat_runtime_settings import resolve_chat_runtime_settings

app = FastAPI(title="IndexGPT API")


def _extract_semantic_tokens(text: str):
    if not text:
        return []
    en = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)]
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    stop_en = {
        "the", "and", "for", "with", "that", "this", "from", "are", "you",
        "your", "was", "were", "have", "has", "had", "will", "would", "can",
        "could", "should", "about", "what", "when", "where", "which",
    }
    stop_zh = {"这个", "那个", "我们", "你们", "他们", "以及", "或者", "如果", "因为", "所以"}
    out = []
    for token in en:
        if token not in stop_en:
            out.append(token)
    for token in zh:
        if token not in stop_zh:
            out.append(token)
    return out


def should_show_doc_sources(context_hits, context_items, answer_text: str) -> bool:
    """
    Heuristic confidence check for whether document citations are truly supporting
    the current answer.
    """
    if not context_items:
        return False

    scores = []
    for item in context_hits or []:
        try:
            scores.append(float(item.get("score", 0.0)))
        except Exception:
            continue
    if not scores:
        return False

    top_scores = sorted(scores, reverse=True)[:3]
    max_score = top_scores[0]
    avg_top = sum(top_scores) / len(top_scores)

    pages = []
    for item in context_items:
        try:
            pages.append(int(item.get("page", 0)))
        except Exception:
            continue
    all_first_page = bool(pages) and all(p == 1 for p in pages)

    context_text = "\n".join((item.get("text") or "")[:1500] for item in context_items)
    answer_tokens = _extract_semantic_tokens(answer_text)
    context_tokens = set(_extract_semantic_tokens(context_text))
    overlap = 0
    if answer_tokens and context_tokens:
        overlap = sum(1 for t in answer_tokens if t in context_tokens)
    grounding_ratio = (overlap / len(answer_tokens)) if answer_tokens else 0.0

    # Hard rejection: retrieval itself is weak.
    if max_score < 0.30 and avg_top < 0.26:
        return False

    # If all refs point to page 1, require stronger grounding evidence.
    if all_first_page and max_score < 0.58 and avg_top < 0.45 and grounding_ratio < 0.18:
        return False

    # Generic grounding guard.
    if grounding_ratio < 0.08 and max_score < 0.52:
        return False

    return True


def fetch_web_snippets(query: str, max_results: int = 3):
    """
    Lightweight web snippets via DuckDuckGo Instant Answer API.
    Returns items shaped like RAG context entries for model consumption.
    """
    if not query.strip():
        return []

    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1,
        "skip_disambig": 1,
    })
    url = f"https://api.duckduckgo.com/?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        payload = {}

    snippets = []
    seen_links = set()

    def append_item(title: str, text: str, link: str):
        content = (text or "").strip()
        if not content:
            return
        norm_link = (link or "").strip()
        if norm_link and norm_link in seen_links:
            return
        if norm_link:
            seen_links.add(norm_link)
        snippets.append({
            "source": f"[Web] {title or 'DuckDuckGo'}",
            "page": 1,
            "score": 0.0,
            "text": content + (f"\nURL: {norm_link}" if norm_link else "")
        })

    def append_from_feed(feed_url: str, source_label: str, limit: int):
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_text = resp.read().decode("utf-8", errors="ignore")
            root = ET.fromstring(xml_text)
        except Exception:
            return

        count = 0
        items = root.findall(".//item")
        for item in items:
            if count >= limit:
                break
            title = (item.findtext("title") or "").strip()
            desc = re.sub(r"<[^>]+>", "", (item.findtext("description") or "")).strip()
            link = (item.findtext("link") or "").strip()
            append_item(f"{source_label} {title}".strip(), desc or title, link)
            count += 1

        if count >= limit:
            return
        entries = root.findall(".//{*}entry")
        for entry in entries:
            if count >= limit:
                break
            title = (entry.findtext(".//{*}title") or "").strip()
            summary = (
                entry.findtext(".//{*}summary")
                or entry.findtext(".//{*}content")
                or ""
            )
            summary = re.sub(r"<[^>]+>", "", summary).strip()
            link = ""
            link_el = entry.find(".//{*}link")
            if link_el is not None:
                link = (link_el.attrib.get("href") or "").strip() or (link_el.text or "").strip()
            append_item(f"{source_label} {title}".strip(), summary or title, link)
            count += 1

    abstract_text = payload.get("AbstractText") or ""
    if abstract_text:
        append_item(payload.get("Heading") or "DuckDuckGo", abstract_text, payload.get("AbstractURL") or "")

    def walk_topics(items):
        for item in items or []:
            if isinstance(item, dict) and item.get("Topics"):
                walk_topics(item.get("Topics"))
                continue
            if not isinstance(item, dict):
                continue
            append_item("DuckDuckGo", item.get("Text") or "", item.get("FirstURL") or "")

    walk_topics(payload.get("RelatedTopics") or [])

    if not snippets:
        google_rss_url = (
            "https://news.google.com/rss/search?"
            + urllib.parse.urlencode({
                "q": query,
                "hl": "zh-CN",
                "gl": "CN",
                "ceid": "CN:zh-Hans",
            })
        )
        append_from_feed(google_rss_url, "Google News", max(1, int(max_results)))

    if not snippets:
        bing_rss_url = (
            "https://www.bing.com/news/search?"
            + urllib.parse.urlencode({
                "q": query,
                "format": "rss",
                "setlang": "zh-cn",
            })
        )
        append_from_feed(bing_rss_url, "Bing News", max(1, int(max_results)))

    if not snippets:
        # China-friendly AI media RSS fallbacks.
        ai_feeds = [
            ("https://www.jiqizhixin.com/rss", "机器之心"),
            ("https://www.qbitai.com/feed", "量子位"),
            ("https://www.leiphone.com/category/ai/feed", "雷锋网 AI"),
        ]
        each_limit = max(1, int(max_results))
        for feed_url, label in ai_feeds:
            append_from_feed(feed_url, label, each_limit)
            if len(snippets) >= max_results:
                break

    return snippets[:max(1, int(max_results))]

# Global state to track background tasks
class TaskManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.task_type: Optional[str] = None

    def is_running(self):
        if self.process and self.process.poll() is None:
            return True
        self.process = None
        self.task_type = None
        return False

    def start_task(self, task_type: str, cmd: list, log_path: Path):
        if self.is_running():
            return False

        log_file = open(log_path, "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        self.task_type = task_type
        return True

    def stop_task(self):
        if self.is_running():
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process = None
            self.task_type = None
            return True
        return False

task_manager = TaskManager()
resource_monitor = ResourceMonitor()

# Local keyword translation model (lazy loaded)
keyword_model = None
keyword_tokenizer = None
keyword_model_id = None
keyword_model_lock = threading.Lock()


def get_keyword_model(model_id: str):
    global keyword_model, keyword_tokenizer, keyword_model_id
    with keyword_model_lock:
        if keyword_model is not None and keyword_tokenizer is not None and keyword_model_id == model_id:
            return keyword_tokenizer, keyword_model

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        model.eval()

        keyword_model = model
        keyword_tokenizer = tokenizer
        keyword_model_id = model_id
        return keyword_tokenizer, keyword_model


def parse_keywords(raw: str):
    content = (raw or "").strip()
    if not content:
        return []

    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()

    # Prefer strict JSON array first.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if not isinstance(item, str):
                    continue
                token = item.strip().lower()
                if token and re.search(r"[a-z]", token):
                    out.append(token)
            return list(dict.fromkeys(out))[:12]
    except Exception:
        pass

    # Fallback: try extracting bracketed array text.
    arr_match = re.search(r"\[[\s\S]*\]", content)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group(0))
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    if not isinstance(item, str):
                        continue
                    token = item.strip().lower()
                    if token and re.search(r"[a-z]", token):
                        out.append(token)
                return list(dict.fromkeys(out))[:12]
        except Exception:
            pass

    # Last fallback: split by commas/newlines.
    rough = re.split(r"[,;\n]+", content)
    out = []
    for item in rough:
        token = re.sub(r"[^a-zA-Z0-9 _-]", "", item).strip().lower()
        if token and re.search(r"[a-z]", token):
            out.append(token)
    return list(dict.fromkeys(out))[:12]

# Allow CORS for frontend access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/papers")
async def list_papers():
    papers = []
    for p in PAPERS_DIR.glob("*.pdf"):
        papers.append({
            "name": p.name,
            "size": f"{os.path.getsize(p) / 1024 / 1024:.2f} MB",
            "path": str(p)
        })
    return papers

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    file_path = PAPERS_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "success"}

@app.get("/api/papers/{filename}")
async def get_paper(filename: str):
    file_path = PAPERS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='application/pdf')

@app.delete("/api/papers/{filename}")
async def delete_paper(filename: str):
    file_path = PAPERS_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"status": "deleted"}
    return {"status": "not_found"}

@app.post("/api/index")
async def rebuild_index():
    if task_manager.start_task("Indexing", ["python", "-u", "scripts/a_rag.py"], TRAIN_LOG_PATH):
        return {"status": "indexing_started"}
    return {"status": "error", "message": f"Task {task_manager.task_type} is already running"}, 400

@app.post("/api/train")
async def start_training():
    if task_manager.start_task("Training", ["python", "-u", "scripts/c_train_lora.py"], TRAIN_LOG_PATH):
        return {"status": "training_started"}
    return {"status": "error", "message": f"Task {task_manager.task_type} is already running"}, 400

@app.post("/api/generate-sft")
async def generate_sft():
    if task_manager.start_task("SFT Generation", ["python", "-u", "scripts/b_generate_sft.py"], TRAIN_LOG_PATH):
        return {"status": "sft_generation_started"}
    return {"status": "error", "message": f"Task {task_manager.task_type} is already running"}, 400

@app.get("/api/status")
async def get_status():
    return {
        "is_running": task_manager.is_running(),
        "task_type": task_manager.task_type
    }

@app.post("/api/stop")
async def stop_task():
    if task_manager.stop_task():
        return {"status": "stopped"}
    return {"status": "error", "message": "No task running"}, 400

@app.get("/api/logs")
async def get_logs():
    if not TRAIN_LOG_PATH.exists():
        return {"logs": "No logs yet."}
    with open(TRAIN_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        return {"logs": f.read()[-5000:]}

@app.get("/api/resources/live")
async def get_live_resources():
    snapshot = resource_monitor.collect()
    return asdict(snapshot)

@app.post("/api/validate/path")
async def validate_path(data: dict):
    path_str = data.get("path", "")
    if not path_str:
        return {"valid": False, "reason": "Empty path"}
    path = Path(path_str)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    # Check if path exists and is a directory
    is_valid = path.exists() and path.is_dir()
    return {"valid": is_valid}

@app.post("/api/validate/api")
async def validate_api(data: dict):
    api_key = data.get("api_key")
    base_url = data.get("base_url")
    model_id = data.get("model_id")

    if not api_key or not base_url or not model_id:
        return {"valid": False, "reason": "Missing credentials"}

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # Deep check: Attempt a very small completion to force real auth check
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
        return {"valid": True}
    except Exception as e:
        error_msg = str(e)
        # Normalize common errors
        if "401" in error_msg or "Authentication" in error_msg:
            return {"valid": False, "reason": "Invalid API Key"}
        return {"valid": False, "reason": error_msg}

@app.post("/api/highlight-keywords")
async def highlight_keywords(data: dict):
    query = (data.get("query") or "").strip()
    if not query:
        return {"keywords": []}

    # Only do translation/expansion when CJK exists in query.
    if not re.search(r"[\u3400-\u9fff]", query):
        return {"keywords": []}

    try:
        settings = load_settings()
        model_id = settings.get("BASE_MODEL_ID") or BASE_MODEL_ID
        tokenizer, model = get_keyword_model(model_id)

        prompt = (
            "You are a multilingual keyword extraction assistant.\n"
            "Task: Convert the user query to 5-10 concise English technical keywords for document highlighting.\n"
            "Output rules:\n"
            "- Output ONLY a JSON array of lowercase strings.\n"
            "- No explanation, no markdown.\n"
            "- Keep terms domain-agnostic and specific.\n\n"
            f"Query: {query}"
        )

        messages = [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_ids = output[0][inputs["input_ids"].shape[-1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return {"keywords": parse_keywords(raw)}
    except Exception:
        return {"keywords": []}

@app.get("/api/settings")
async def get_settings():
    return load_settings()

@app.get("/api/available-models")
async def list_available_models():
    """Scan runtime model directory for subfolders and return them as choices."""
    models_dir = RAG_INDEX_DIR / "models"
    if not models_dir.exists():
        return {"llm": [], "embedding": []}

    # Simple heuristic: LLM usually contains 'Qwen' or 'Llama', BGE/Embedding usually contains 'bge' or 'embedding'
    # We'll just return all subdirectories for both for now, or user can choose.
    items = [d.name for d in models_dir.iterdir() if d.is_dir()]
    return {
        "all": items,
        "llm_hints": [i for i in items if "bge" not in i.lower()],
        "emb_hints": [i for i in items if "bge" in i.lower()]
    }

@app.post("/api/settings")
async def save_settings(data: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Note: In a production environment, you might need to re-initialize
        # certain global variables, but for this architecture,
        # scripts load config.py on execution.
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for p in CHATS_DIR.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "id": p.stem,
                    "title": data.get("title", "Untitled Chat"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": p.stat().st_mtime
                })
        except:
            continue
    # Sort by latest file modification time (most recently updated first).
    return sorted(sessions, key=lambda x: x.get("updated_at", 0), reverse=True)

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    file_path = CHATS_DIR / f"{session_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/sessions")
async def create_session():
    session_id = str(uuid.uuid4())
    file_path = CHATS_DIR / f"{session_id}.json"
    data = {
        "id": session_id,
        "title": "New Chat",
        "history": [],
        "created_at": str(asyncio.get_event_loop().time()) # Simplified timestamp
    }
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"id": session_id}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    file_path = CHATS_DIR / f"{session_id}.json"
    if file_path.exists():
        file_path.unlink()
        return {"status": "deleted"}
    return {"status": "not_found"}

@app.post("/api/chat")
async def chat(data: dict):
    if task_manager.is_running():
        raise HTTPException(
            status_code=400,
            detail=f"Cannot chat while {task_manager.task_type} is running. Please wait for it to finish."
        )
    query = data.get("query")
    session_id = data.get("session_id")
    compare_mode = bool(data.get("compare_mode"))
    web_search = bool(data.get("web_search"))

    history = []
    if session_id:
        file_path = CHATS_DIR / f"{session_id}.json"
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                session_data = json.load(f)
                history = [{"role": m["role"], "content": m["content"]} for m in session_data.get("history", [])]

    runtime = resolve_chat_runtime_settings(load_settings())
    chat_topk = runtime["chat_topk"]
    compare_topk = runtime["compare_topk"]
    chat_max_tokens = runtime["chat_max_tokens"]
    chat_history_rounds = runtime["chat_history_rounds"]

    context_hits = rag_query(query, topk=compare_topk if compare_mode else chat_topk)
    context = select_compare_sources(context_hits, max_papers=5) if compare_mode else context_hits
    web_context = fetch_web_snippets(query, max_results=3) if web_search else []
    context_for_model = context + web_context
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    web_guidance = ""
    if web_search:
        if web_context:
            web_guidance = (
                f"\n\n[Web mode enabled]\n"
                f"Current server time: {now_str}\n"
                "Web snippets are provided in context with source tag [Web]. "
                "Use them directly when answering current-date/current-news questions. "
                "Do not claim you cannot access the internet. "
                "If snippets are not enough, explicitly say snippets are insufficient."
            )
        else:
            web_guidance = (
                f"\n\n[Web mode enabled, but no usable web snippets were returned]\n"
                f"Current server time: {now_str}\n"
                "Do not claim generic no-internet access; state that this specific retrieval returned no usable snippets."
            )
    query_for_model = f"{query}{web_guidance}"

    stop_event = threading.Event()

    async def event_generator():
        full_answer = ""
        if compare_mode:
            compare_query = build_compare_query(query_for_model)
            full_answer = (
                run_inference(compare_query, context_for_model, max_tokens=chat_max_tokens)
                if context_for_model
                else "No relevant papers found for comparison."
            )
            for i in range(0, len(full_answer), 40):
                token = full_answer[i:i + 40]
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.01)
        else:
            streamer = run_inference_stream(
                query_for_model,
                context_for_model,
                history=history,
                max_tokens=chat_max_tokens,
                stop_event=stop_event,
                history_rounds=chat_history_rounds,
            )
            try:
                for token in streamer:
                    full_answer += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                print("Chat request cancelled by user.")
                stop_event.set()
                raise

        sources_payload = context if should_show_doc_sources(context_hits, context, full_answer) else None
        yield f"data: {json.dumps({'sources': sources_payload})}\n\n"

        if session_id:
            file_path = CHATS_DIR / f"{session_id}.json"
            if file_path.exists():
                with file_path.open("r", encoding="utf-8") as f:
                    session_data = json.load(f)

                if not session_data["history"]:
                    session_data["title"] = query[:30] + ("..." if len(query) > 30 else "")

                session_data["history"].append({"role": "user", "content": query})
                session_data["history"].append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources_payload,
                    "compare_mode": compare_mode,
                    "web_search": web_search
                })

                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT)
