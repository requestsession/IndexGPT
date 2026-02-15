# IndexGPT

[English](./README.md)

IndexGPT 是一个面向科研场景的本地 AI 助手工作流，围绕 **RAG + SFT + LoRA** 构建。
它将论文导入、检索问答、SFT 数据生成、LoRA 训练和网页端聊天整合在同一个项目中。

## 0. 项目标识

- 仓库地址：`https://github.com/requestsession/IndexGPT`
- 维护者：`requestsession`

## 1. 环境要求

- Python: 3.10+
- Node.js: 18+
- 操作系统：Windows / Linux（当前仓库对 Windows 更友好）
- GPU：建议用于推理与训练

安装依赖：

```bash
pip install -r requirements.txt
cd web/frontend
npm install
```

## 2. 模型放置（重要）

本项目使用**本地模型目录**。模型不是单一文件，而是一个包含多文件的目录，例如：

- `config.json`
- tokenizer 文件（`tokenizer.json`, `tokenizer_config.json`, ...）
- 权重文件（`*.safetensors` / `pytorch_model*.bin`）

### 推荐目录

请将模型目录放在：

```text
workspace/rag_index/models/
```

示例：

```text
workspace/rag_index/models/
  Qwen2.5-7B-Instruct/
  bge-base-en-v1.5/
```

然后在系统设置页面中选择：

- `Base LLM`
- `Embedding Model`

后端会扫描 `workspace/rag_index/models/` 并返回可选模型目录。

## 3. 启动项目

在项目根目录执行：

```bash
python web/backend/main.py
```

另开一个终端：

```bash
cd web/frontend
npm run dev -- --host
```

默认端口：

- 后端：`8176`（来自 `config.py` / 设置页）
- 前端：Vite 默认 `5173`

## 4. 界面演示

### AI 助手

![AI 助手](./web/showcase/figs/1_AI_Assistant.png)

### 文献库

![文献库](./web/showcase/figs/2_Documents.png)

### 模型训练

![模型训练](./web/showcase/figs/3_Training_Lab.png)

### 系统设置

![系统设置](./web/showcase/figs/4_System_Settings.png)

## 5. 端到端流程

### A. 上传论文

- 打开 `Documents` 页面
- 上传 PDF 文件
- 文件存储在 `workspace/papers/`

### B. 构建 RAG 索引

- 打开 `Training Lab`
- 点击 **RAG Indexing -> Execute**

生成产物：

- `workspace/rag_index/parse/*.json`
- `workspace/rag_index/faiss.index`
- `workspace/rag_index/texts.json`
- `workspace/rag_index/meta.json`

### C. 生成 SFT 数据

- 点击 **SFT Generation -> Execute**
- 输出文件：`workspace/data/sft.jsonl`

### D. 训练 LoRA

- 点击 **LoRA Training -> Execute**
- 适配器输出路径：`workspace/outputs/lora/`

### E. 聊天问答

- 打开 `AI Assistant`
- 基于已索引论文提问
- 可选模式：
  - Compare 模式（跨论文对比）
  - Web 模式（互联网片段）

## 6. 等价 CLI 命令

在项目根目录运行：

```bash
python -u scripts/a_rag.py
python -u scripts/b_generate_sft.py
python -u scripts/c_train_lora.py
```

推理入口：

```bash
python scripts/d_infer_lora.py
```

## 7. 关键目录

- `workspace/papers/`: 上传的 PDF
- `workspace/rag_index/`: 检索索引及元数据
- `workspace/data/sft.jsonl`: SFT 数据集
- `workspace/outputs/lora/`: LoRA 训练输出
- `workspace/logs/`: 运行日志
- `workspace/chats/`: 对话持久化

## 8. 运行说明

- 重任务互斥（索引 / SFT / 训练不可并行）
- 重任务运行期间聊天会被禁用
- 历史对话窗口由设置项 `CHAT_HISTORY_ROUNDS` 控制
- 如果 `workspace/` 下运行目录被删除，加载 `config.py` 时会自动重建

## 9. 常见问题

### 设置页看不到模型

检查：

1. 模型目录是否存在于 `workspace/rag_index/models/`
2. 模型目录是否完整（而非仅有单个权重分片）
3. 刷新设置页或重启后端

### 聊天没有来源引用

通常表示检索置信度不足。请检查：

1. PDF 是否为可提取文本（不是纯扫描图像）
2. 新增 PDF 后是否重建索引
3. 提问是否足够具体

### 训练很快失败

检查：

1. `Base LLM` 路径是否有效
2. `workspace/data/sft.jsonl` 是否存在
3. GPU/显存与依赖版本是否匹配

## 10. 引用

如果本项目对你的工作有帮助，可使用以下格式引用：

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
