import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { splitHighlightedText } from './utils/evidence';
import { shouldAutoScroll } from './utils/chatScroll';
import { parseResourceLogs, summarizeResourcePeaks } from './utils/resourceMetrics';
import { pickDisplaySeries } from './utils/resourceLive';
import {
  FileText,
  Settings as SettingsIcon,
  MessageSquare,
  Upload,
  Trash2,
  RefreshCw,
  Play,
  Database,
  Search,
  ChevronRight,
  Terminal,
  Eye,
  EyeOff,
  Sun,
  Moon,
  Languages
} from 'lucide-react';

const API_BASE = '/api';

const detectSystemLang = () => {
  if (typeof navigator === 'undefined') return 'en';
  const candidates = Array.isArray(navigator.languages) && navigator.languages.length
    ? navigator.languages
    : [navigator.language || 'en'];
  const hasChinese = candidates.some((item) => String(item || '').toLowerCase().startsWith('zh'));
  return hasChinese ? 'zh' : 'en';
};

const translations = {
  en: {
    projectName: "Project Name",
    documents: "Documents",
    training: "Training Lab",
    chat: "AI Assistant",
    settings: "System Settings",
    upload: "Upload PDF",
    newChat: "New Chat",
    thinking: "AI is thinking...",
    askPlaceholder: "Ask about your papers...",
    send: "Send",
    compareMode: "Compare",
    modes: "Modes",
    compareHint: "Cross-paper comparison mode",
    webSearch: "Web",
    webSearchHint: "Enable internet snippets",
    sources: "Sources Context",
    evidence: "Evidence Paragraph",
    evidenceHint: "Highlighted with question keywords (includes translated terms when needed).",
    preview: "PDF Preview",
    backToList: "Back to list",
    indexing: "RAG Indexing",
    indexingDesc: "Parse PDFs and update vector database.",
    sft: "SFT Generation",
    sftDesc: "Create instruction-tuning data from papers.",
    train: "LoRA Training",
    trainDesc: "Start asynchronous fine-tuning process.",
    stop: "Stop",
    execute: "Execute",
    saveSettings: "Save System Settings",
    coreConfig: "Core Configuration (Restart Required)",
    runtimeConfig: "Runtime & API Configuration (OpenAI Compatible)",
    backendPortLabel: "Backend Port",
    baseLlmLabel: "Base LLM",
    embeddingModelLabel: "Embedding Model",
    researchFieldLabel: "Research Field",
    chatTopKLabel: "Chat Retrieval TopK",
    compareTopKLabel: "Compare Mode TopK",
    chatMaxTokensLabel: "Answer Max Tokens",
    chatHistoryRoundsLabel: "History Rounds",
    sftModelSource: "SFT Generation Mode",
    localModel: "Local Model (Private)",
    apiModel: "Cloud API (Fast)",
    selectModel: "Select a model...",
    noModelsFound: "No models found in workspace/rag_index/models/",
    restartNote: "* Changing these values requires manual restart of 'python web/backend/main.py'.",
    saveSuccess: "Settings saved successfully!\n\nNOTE: If you changed Port or Model Paths, you MUST restart the backend server manually.",
    saveError: "Failed to save settings.",
    deleteConfirm: "Are you sure you want to delete this?",
    noPapers: "No papers yet. Upload some PDFs to get started!",
    unsavedChanges: "You have unsaved changes. Do you want to leave? Your changes will be discarded.",
    searchPlaceholder: "Search papers by name...",
    metricsTitle: "Resource Monitor",
    cpuLabel: "CPU",
    ramLabel: "RAM",
    gpuLabel: "GPU",
    peakLabel: "Peak",
    samplesLabel: "Latest {count} samples",
    noSamplesLabel: "No resource samples yet"
  },
  zh: {
    projectName: "项目名称",
    documents: "文献库",
    training: "模型训练",
    chat: "AI 助手",
    settings: "系统设置",
    upload: "上传 PDF",
    newChat: "新建对话",
    thinking: "AI 思考中...",
    askPlaceholder: "询问关于文献的问题...",
    send: "发送",
    compareMode: "对比模式",
    modes: "模式",
    compareHint: "跨论文对比回答",
    webSearch: "联网",
    webSearchHint: "启用互联网检索片段",
    sources: "引用来源",
    evidence: "证据段落",
    evidenceHint: "按问题关键词高亮（必要时自动补充翻译关键词）",
    preview: "PDF 预览",
    backToList: "返回列表",
    indexing: "RAG 索引",
    indexingDesc: "解析 PDF 并更新向量数据库。",
    sft: "SFT 数据生成",
    sftDesc: "基于文献创建指令微调数据。",
    train: "LoRA 训练",
    trainDesc: "启动异步模型微调流程。",
    stop: "停止",
    execute: "执行",
    saveSettings: "保存系统设置",
    coreConfig: "核心配置（需重启）",
    runtimeConfig: "运行环境与 API 配置（OpenAI 兼容）",
    backendPortLabel: "后端端口",
    baseLlmLabel: "基础大模型",
    embeddingModelLabel: "向量模型",
    researchFieldLabel: "研究领域",
    chatTopKLabel: "聊天检索 TopK",
    compareTopKLabel: "对比模式 TopK",
    chatMaxTokensLabel: "回答最大 Token",
    chatHistoryRoundsLabel: "历史轮数",
    sftModelSource: "SFT 数据生成模式",
    localModel: "本地模型（隐私优先）",
    apiModel: "云端 API（速度优先）",
    selectModel: "选择模型...",
    noModelsFound: "未在 workspace/rag_index/models/ 下找到模型",
    restartNote: "* 修改这些值后需手动重启 'python web/backend/main.py'。",
    saveSuccess: "设置保存成功！\n\n注意：如果修改了端口或模型路径，必须手动重启后端服务。",
    saveError: "设置保存失败。",
    deleteConfirm: "确定要删除吗？",
    noPapers: "暂无文献。上传 PDF 开始使用吧！",
    unsavedChanges: "您有未保存的设置。确定要离开吗？未保存内容将会丢失。",
    searchPlaceholder: "搜索文献名称...",
    metricsTitle: "资源监控",
    cpuLabel: "CPU",
    ramLabel: "内存",
    gpuLabel: "GPU",
    peakLabel: "峰值",
    samplesLabel: "最近 {count} 个采样点",
    noSamplesLabel: "暂无资源采样数据"
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [lang, setLang] = useState(() => detectSystemLang());
  const [darkMode, setDarkMode] = useState(false);
  const [papers, setPapers] = useState([]);
  const [paperSearch, setPaperSearch] = useState("");
  const [logs, setLogs] = useState("等待日志...");
  const [query, setQuery] = useState("");
  const [compareMode, setCompareMode] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [debugSources, setDebugSources] = useState([]);
  const [activeTask, setActiveTask] = useState({ is_running: false, task_type: null });
  const [sessions, setSessions] = useState([]);
  const [liveResourceSeries, setLiveResourceSeries] = useState([]);
  const liveStepRef = useRef(0);
  const abortControllerRef = useRef(null);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [previewFile, setPreviewFile] = useState(null); // { name, page }
  const [selectedEvidence, setSelectedEvidence] = useState(null); // { source, page, text, score, query }
  const [paperPage, setPaperPage] = useState(1);
  const papersPerPage = 9;
  const [sidebarWidth, setSidebarWidth] = useState(320);
  const [isResizing, setIsResizing] = useState(false);
  const [validation, setValidation] = useState({
    BASE_MODEL_ID: null,
    BGE_MODEL_PATH: null,
    API: null
  });
  const [settings, setSettings] = useState({
    PROJECT_NAME: "IndexGPT",
    RESEARCH_FIELD: "",
    BACKEND_PORT: 8176,
    BASE_MODEL_ID: "",
    BGE_MODEL_PATH: "",
    API_KEY: "",
    API_BASE_URL: "",
    API_MODEL_ID: "",
    CHAT_TOPK: 3,
    COMPARE_TOPK: 12,
    CHAT_MAX_TOKENS: 1200,
    CHAT_HISTORY_ROUNDS: 3,
    SFT_MODEL_SOURCE: "local"
  });
  const [lastSavedSettings, setLastSavedSettings] = useState(null);
  const [availableModels, setAvailableModels] = useState({ all: [], llm_hints: [], emb_hints: [] });
  const validationRequestRef = useRef({
    BASE_MODEL_ID: "",
    BGE_MODEL_PATH: "",
    API: ""
  });
  const apiValidationTimerRef = useRef(null);

  const messagesEndRef = useRef(null);
  const chatScrollRef = useRef(null);
  const modeMenuRef = useRef(null);
  const [stickToBottom, setStickToBottom] = useState(true);

  const handleTabChange = (newTab) => {
    if (activeTab === 'settings' && newTab !== 'settings') {
      const hasChanges = JSON.stringify(settings) !== JSON.stringify(lastSavedSettings);
      if (hasChanges) {
        if (window.confirm(t.unsavedChanges)) {
          const restored = lastSavedSettings || settings;
          setSettings(restored); // 杩樺師
          validationRequestRef.current.BASE_MODEL_ID = restored.BASE_MODEL_ID || "";
          validationRequestRef.current.BGE_MODEL_PATH = restored.BGE_MODEL_PATH || "";
          validationRequestRef.current.API = `${restored.API_KEY || ""}|${restored.API_BASE_URL || ""}|${restored.API_MODEL_ID || ""}`;
          validateAll(restored);
          setActiveTab(newTab);
        }
        return;
      }
    }
    setActiveTab(newTab);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (activeTab === 'chat' && stickToBottom) {
      scrollToBottom();
    }
  }, [chatHistory, loading, activeTab, stickToBottom]);

  const handleChatScroll = () => {
    if (!chatScrollRef.current) return;
    setStickToBottom(shouldAutoScroll(chatScrollRef.current, 80));
  };

  useEffect(() => {
    const handleOutsideClick = (event) => {
      if (!modeMenuRef.current) return;
      if (!modeMenuRef.current.contains(event.target)) {
        setModeMenuOpen(false);
      }
    };
    window.addEventListener('mousedown', handleOutsideClick);
    return () => window.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  const t = translations[lang];
  const l10n = lang === 'zh'
    ? {
        apiModelIdLabel: "API 模型 ID",
        apiBaseUrlLabel: "API 地址",
        apiKeyLabel: "API 密钥",
        pdfViewerTitle: "PDF 预览",
        uploadPdfOnlyAlert: "仅支持上传 PDF 文件。",
        uploadSuccessAlert: "成功上传 {count} 个文件。",
        deleteFileFailedAlert: "删除文件失败。",
        trainingStartedAlert: "训练已启动。",
        trainingStartFailedAlert: "启动训练失败。",
        sftStartedAlert: "SFT 数据生成已启动。",
        sftStartFailedAlert: "启动 SFT 数据生成失败。",
        indexingStartedAlert: "索引构建已启动。",
        indexingStartFailedAlert: "启动索引构建失败。",
        taskStoppedAlert: "任务已停止。",
        sendFailedMessage: "发送消息失败",
        stopLabel: "停止",
        executeLabel: "执行",
        validLabel: "有效",
        invalidLabel: "无效",
        trainingLogLabel: "训练日志"
      }
    : {
        apiModelIdLabel: "API Model ID",
        apiBaseUrlLabel: "API Base URL",
        apiKeyLabel: "API Key",
        pdfViewerTitle: "PDF Viewer",
        uploadPdfOnlyAlert: "Only PDF files are allowed.",
        uploadSuccessAlert: "Successfully uploaded {count} file(s).",
        deleteFileFailedAlert: "Failed to delete file.",
        trainingStartedAlert: "Training started!",
        trainingStartFailedAlert: "Failed to start training",
        sftStartedAlert: "SFT Generation started!",
        sftStartFailedAlert: "Failed to start SFT generation",
        indexingStartedAlert: "Indexing started!",
        indexingStartFailedAlert: "Failed to start indexing",
        taskStoppedAlert: "Task stopped successfully.",
        sendFailedMessage: "Failed to send message",
        stopLabel: "Stop",
        executeLabel: "Execute",
        validLabel: "VALID",
        invalidLabel: "INVALID",
        trainingLogLabel: "training_output.log"
      };
  const logResourcePoints = parseResourceLogs(logs).slice(-60);
  const resourcePoints = pickDisplaySeries(liveResourceSeries, logResourcePoints);
  const resourceLatest = resourcePoints.length ? resourcePoints[resourcePoints.length - 1] : null;
  const resourcePeaks = summarizeResourcePeaks(resourcePoints);

  const getLatestUserQuery = (history) => {
    const lastUser = [...history].reverse().find((msg) => msg.role === 'user');
    return lastUser?.content || "";
  };

  const getNearestUserQuery = (history, assistantIndex) => {
    for (let idx = assistantIndex - 1; idx >= 0; idx -= 1) {
      if (history[idx]?.role === 'user') return history[idx].content || "";
    }
    return getLatestUserQuery(history);
  };

  const containsCJK = (text) => /[\u3400-\u9fff]/.test(text || "");

  const expandQueryForHighlight = async (rawQuery) => {
    if (!containsCJK(rawQuery)) return rawQuery;

    try {
      const res = await fetch(`${API_BASE}/highlight-keywords`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: rawQuery })
      });
      if (!res.ok) return rawQuery;

      const data = await res.json();
      const keywords = Array.isArray(data.keywords) ? data.keywords.filter(Boolean) : [];
      if (!keywords.length) return rawQuery;

      return `${rawQuery} ${keywords.join(" ")}`.trim();
    } catch (_) {
      return rawQuery;
    }
  };

  const openEvidence = async (source, queryText = "") => {
    const baseQuery = queryText || getLatestUserQuery(chatHistory);
    setPreviewFile({ name: source.source, page: source.page });
    setSelectedEvidence({
      ...source,
      query: baseQuery
    });

    const expandedQuery = await expandQueryForHighlight(baseQuery);
    if (expandedQuery === baseQuery) return;

    setSelectedEvidence((prev) => {
      if (!prev) return prev;
      if (prev.source !== source.source || prev.page !== source.page || prev.text !== source.text) {
        return prev;
      }
      return { ...prev, query: expandedQuery };
    });
  };

  // Handle sidebar resizing
  const startResizing = (e) => {
    setIsResizing(true);
    e.preventDefault();
  };

  const stopResizing = () => {
    setIsResizing(false);
  };

  const resize = (e) => {
    if (isResizing) {
      const newWidth = window.innerWidth - e.clientX - 32;
      if (newWidth > 200 && newWidth < 800) {
        setSidebarWidth(newWidth);
      }
    }
  };

  useEffect(() => {
    window.addEventListener('mousemove', resize);
    window.addEventListener('mouseup', stopResizing);
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [isResizing]);

  // Fetch papers
  const fetchPapers = async () => {
    try {
      const res = await fetch(`${API_BASE}/papers`);
      const data = await res.json();
      setPapers(data);
    } catch (e) { console.error("Failed to fetch papers", e); }
  };

  // Fetch sessions
  const fetchSessions = async (selectLatest = false) => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data = await res.json();
      setSessions(data);
      if (selectLatest && data.length > 0) {
        handleSelectSession(data[0].id);
      } else if (data.length === 0 && activeTab === 'chat') {
        handleNewChat();
      }
    } catch (e) { console.error("Failed to fetch sessions", e); }
  };

  // Fetch settings
  const fetchSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/settings`);
      const data = await res.json();
      setSettings(data);
      setLastSavedSettings(data);
      if (data.PROJECT_NAME) {
        document.title = data.PROJECT_NAME;
      }
      // Trigger validation on load
      validateAll(data);
    } catch (e) { console.error("Failed to fetch settings", e); }

    // Also fetch available models
    try {
      const res = await fetch(`${API_BASE}/available-models`);
      const data = await res.json();
      setAvailableModels(data);
    } catch (e) { console.error("Failed to fetch models", e); }
  };

  const validatePathValue = async (path) => {
    const res = await fetch(`${API_BASE}/validate/path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path })
    });
    const data = await res.json();
    return data.valid;
  };

  const validateApiValue = async (s) => {
    if (!s.API_KEY || !s.API_BASE_URL || !s.API_MODEL_ID) return false;
    const res = await fetch(`${API_BASE}/validate/api`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: s.API_KEY,
        base_url: s.API_BASE_URL,
        model_id: s.API_MODEL_ID
      })
    });
    const data = await res.json();
    return data.valid;
  };

  const validatePathField = async (field, path) => {
    validationRequestRef.current[field] = path;
    try {
      const valid = await validatePathValue(path);
      if (validationRequestRef.current[field] !== path) return;
      setValidation((prev) => ({ ...prev, [field]: valid }));
    } catch (_) {
      if (validationRequestRef.current[field] !== path) return;
      setValidation((prev) => ({ ...prev, [field]: false }));
    }
  };

  const validateApiField = async (s) => {
    const signature = `${s.API_KEY || ""}|${s.API_BASE_URL || ""}|${s.API_MODEL_ID || ""}`;
    validationRequestRef.current.API = signature;
    if (!s.API_KEY || !s.API_BASE_URL || !s.API_MODEL_ID) {
      setValidation((prev) => ({ ...prev, API: false }));
      return;
    }
    setValidation((prev) => ({ ...prev, API: null }));
    try {
      const valid = await validateApiValue(s);
      if (validationRequestRef.current.API !== signature) return;
      setValidation((prev) => ({ ...prev, API: valid }));
    } catch (_) {
      if (validationRequestRef.current.API !== signature) return;
      setValidation((prev) => ({ ...prev, API: false }));
    }
  };

  const validateAll = async (s) => {
    const [vBase, vBge, vApi] = await Promise.all([
      validatePathValue(s.BASE_MODEL_ID),
      validatePathValue(s.BGE_MODEL_PATH),
      validateApiValue(s)
    ]);

    setValidation({
      BASE_MODEL_ID: vBase,
      BGE_MODEL_PATH: vBge,
      API: vApi
    });
  };

  // Poll logs and status every 2s
  useEffect(() => {
    fetchPapers();
    fetchSessions(true);
    fetchSettings();
    const pollRuntimeState = async () => {
      try {
        const logRes = await fetch(`${API_BASE}/logs`);
        const logData = await logRes.json();
        if (logData.logs) setLogs(logData.logs);

        const statusRes = await fetch(`${API_BASE}/status`);
        const statusData = await statusRes.json();
        setActiveTask(statusData);

        const resourceRes = await fetch(`${API_BASE}/resources/live`);
        const resourceData = await resourceRes.json();
        liveStepRef.current += 1;
        setLiveResourceSeries((prev) => {
          const next = [
            ...prev,
            {
              step: liveStepRef.current,
              cpu: resourceData.cpu_percent ?? null,
              ram: resourceData.system_memory_percent ?? null,
              gpu: resourceData.gpu_util_percent ?? null
            }
          ];
          return next.slice(-60);
        });
      } catch (e) {}
    };
    pollRuntimeState();
    const timer = setInterval(pollRuntimeState, 2000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (settings.SFT_MODEL_SOURCE !== 'api') return;
    if (apiValidationTimerRef.current) clearTimeout(apiValidationTimerRef.current);
    apiValidationTimerRef.current = setTimeout(() => {
      validateApiField(settings);
    }, 350);
    return () => {
      if (apiValidationTimerRef.current) clearTimeout(apiValidationTimerRef.current);
    };
  }, [settings.SFT_MODEL_SOURCE, settings.API_KEY, settings.API_BASE_URL, settings.API_MODEL_ID]);

  // Actions
  const handleUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    const nonPdfs = files.filter(f => !f.name.toLowerCase().endsWith('.pdf'));
    if (nonPdfs.length > 0) {
      alert(l10n.uploadPdfOnlyAlert);
      e.target.value = null;
      return;
    }
    let successCount = 0;
    for (let f of files) {
      const formData = new FormData();
      formData.append('file', f);
      try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        if (res.ok) successCount++;
      } catch (err) { console.error("Upload failed for", f.name, err); }
    }
    if (successCount > 0) {
      alert(l10n.uploadSuccessAlert.replace('{count}', String(successCount)));
      fetchPapers();
    }
    e.target.value = null;
  };

  const handleDelete = async (filename) => {
    if (!window.confirm(t.deleteConfirm)) return;
    try {
      const res = await fetch(`${API_BASE}/papers/${filename}`, { method: 'DELETE' });
      if (res.ok) fetchPapers();
      else alert(l10n.deleteFileFailedAlert);
    } catch (e) { console.error("Delete failed", e); }
  };

  const handleTrain = async () => {
    const res = await fetch(`${API_BASE}/train`, { method: 'POST' });
    if (res.ok) alert(l10n.trainingStartedAlert);
    else alert(l10n.trainingStartFailedAlert);
  };

  const handleSFT = async () => {
    const res = await fetch(`${API_BASE}/generate-sft`, { method: 'POST' });
    if (res.ok) alert(l10n.sftStartedAlert);
    else alert(l10n.sftStartFailedAlert);
  };

  const handleIndex = async () => {
    const res = await fetch(`${API_BASE}/index`, { method: 'POST' });
    if (res.ok) alert(l10n.indexingStartedAlert);
    else alert(l10n.indexingStartFailedAlert);
  };

  const handleStop = async () => {
    const res = await fetch(`${API_BASE}/stop`, { method: 'POST' });
    if (res.ok) alert(l10n.taskStoppedAlert);
  };

  const handleStopChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
    }
  };

  const handleNewChat = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`, { method: 'POST' });
      const data = await res.json();
      setChatHistory([]);
      setDebugSources([]);
      setSelectedEvidence(null);
      setPreviewFile(null);
      setCurrentSessionId(data.id);
      setStickToBottom(true);
      fetchSessions();
    } catch (e) { console.error("Failed to create session", e); }
  };

  const handleSelectSession = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${id}`);
      const data = await res.json();
      setCurrentSessionId(id);
      setChatHistory(data.history || []);
      const lastAssistantMsg = [...(data.history || [])].reverse().find(m => m.role === 'assistant');
      setDebugSources(lastAssistantMsg?.sources || []);
      setSelectedEvidence(null);
      setStickToBottom(true);
    } catch (e) { console.error("Failed to load session", e); }
  };

  const handleDeleteSession = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm(t.deleteConfirm)) return;
    try {
      const res = await fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE' });
      if (res.ok) {
        if (currentSessionId === id) {
          setChatHistory([]);
          setCurrentSessionId(null);
          setSelectedEvidence(null);
          setPreviewFile(null);
        }
        fetchSessions();
      }
    } catch (e) { console.error("Failed to delete session", e); }
  };

  const handleSaveSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      if (res.ok) {
        alert(t.saveSuccess);
        fetchSettings();
        // Re-validate after save
        validateAll(settings);
      } else {
        alert(t.saveError);
      }
    } catch (e) {
      console.error("Save error", e);
      alert(t.saveError);
    }
  };

  const handleChat = async () => {
    if (!query) return;

    if (loading && abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let sessionId = currentSessionId;
    if (!sessionId) {
      const res = await fetch(`${API_BASE}/sessions`, { method: 'POST' });
      const data = await res.json();
      sessionId = data.id;
      setCurrentSessionId(sessionId);
    }
    setLoading(true);
    const userMsg = { role: 'user', content: query };
    const assistantMsg = { role: 'assistant', content: "" };
    setChatHistory(prev => [...prev, userMsg, assistantMsg]);
    setSelectedEvidence(null);
    setQuery("");

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId, compare_mode: compareMode, web_search: webSearchEnabled }),
        signal: controller.signal
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || l10n.sendFailedMessage);
      }

      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;
            try {
              const data = JSON.parse(dataStr);
              if (data.sources) {
                setDebugSources(data.sources);
                setChatHistory(prev => {
                  const newHistory = [...prev];
                  if (!newHistory.length) return newHistory;
                  const lastIndex = newHistory.length - 1;
                  newHistory[lastIndex] = {
                    ...newHistory[lastIndex],
                    sources: data.sources
                  };
                  return newHistory;
                });
              } else if (data.token) {
                fullContent += data.token;
                setChatHistory(prev => {
                  const newHistory = [...prev];
                  newHistory[newHistory.length - 1] = {
                    role: 'assistant',
                    content: fullContent,
                    sources: data.sources || newHistory[newHistory.length - 1].sources
                  };
                  return newHistory;
                });
              }
            } catch (e) {}
          }
        }
      }
      fetchSessions();
    } catch (e) {
      setChatHistory(prev => {
        const newHistory = [...prev];
        const errorMsg = e.name === 'AbortError' ? (lang === 'zh' ? '已停止回答' : 'Interrupted') : e.message;
        newHistory[newHistory.length - 1].content = `Error: ${errorMsg}`;
        return newHistory;
      });
    } finally { setLoading(false); }
  };

  return (
    <div className={`flex h-screen font-sans transition-colors duration-300 ${darkMode ? 'dark bg-slate-900 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
      {/* Sidebar */}
      <aside className={`w-64 border-r flex flex-col transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
        <div className={`p-6 border-b ${darkMode ? 'border-slate-700' : 'border-slate-200'}`}>
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Database className="w-6 h-6 text-blue-600" />
            {settings.PROJECT_NAME}
          </h1>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <NavItem icon={<MessageSquare />} label={t.chat} active={activeTab === 'chat'} onClick={() => handleTabChange('chat')} darkMode={darkMode} />
          <NavItem icon={<FileText />} label={t.documents} active={activeTab === 'documents'} onClick={() => handleTabChange('documents')} darkMode={darkMode} />
          <NavItem icon={<RefreshCw />} label={t.training} active={activeTab === 'training'} onClick={() => handleTabChange('training')} darkMode={darkMode} />
          <NavItem icon={<SettingsIcon />} label={t.settings} active={activeTab === 'settings'} onClick={() => handleTabChange('settings')} darkMode={darkMode} />
        </nav>

        <div className={`p-4 border-t flex items-center justify-around ${darkMode ? 'border-slate-700' : 'border-slate-200'}`}>
          <button onClick={() => setLang(lang === 'en' ? 'zh' : 'en')} className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-slate-700 text-slate-300' : 'hover:bg-slate-100 text-slate-600'}`}>
            <Languages size={20} /><span className="text-xs ml-1 font-bold">{lang.toUpperCase()}</span>
          </button>
          <button onClick={() => setDarkMode(!darkMode)} className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-slate-700 text-yellow-400' : 'hover:bg-slate-100 text-slate-600'}`}>
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden">
        <header className={`h-16 border-b flex items-center px-8 justify-between transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
          <div className="flex items-center gap-6 flex-1">
            <h2 className="text-lg font-semibold shrink-0">{t[activeTab]}</h2>
            {activeTab === 'documents' && (
              <div className="relative w-96 max-w-full">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  placeholder={t.searchPlaceholder}
                  value={paperSearch}
                  onChange={(e) => { setPaperSearch(e.target.value); setPaperPage(1); }}
                  className={`w-full pl-10 pr-4 py-1.5 text-sm border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors ${
                    darkMode ? 'bg-slate-900 border-slate-700 text-slate-100' : 'bg-slate-50 border-slate-200 text-slate-900'
                  }`}
                />
              </div>
            )}
          </div>
          <div className="flex gap-4">
            {activeTab === 'documents' && (
              <label className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition">
                <Upload size={18} /><span>{t.upload}</span>
                <input type="file" multiple accept=".pdf" className="hidden" onChange={handleUpload} />
              </label>
            )}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:p-8">
          {activeTab === 'documents' && (
            <div className="flex gap-6 h-full">
              <div className="flex-1 flex flex-col gap-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {(() => {
                    const filteredPapers = papers.filter(p => p.name.toLowerCase().includes(paperSearch.toLowerCase()));
                    const pagePapers = filteredPapers.slice((paperPage - 1) * papersPerPage, paperPage * papersPerPage);

                    return pagePapers.length > 0 ? pagePapers.map((paper) => (
                      <div
                        key={paper.name}
                        onClick={() => setPreviewFile({ name: paper.name, page: 1 })}
                        className={`p-6 rounded-xl border shadow-sm hover:shadow-md transition cursor-pointer group ${
                          previewFile?.name === paper.name
                            ? (darkMode ? 'border-blue-500 bg-blue-900/10' : 'border-blue-400 bg-blue-50')
                            : (darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200')
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className={`p-3 rounded-lg ${darkMode ? 'bg-blue-900/30 text-blue-400' : 'bg-blue-50 text-blue-600'}`}>
                            <FileText size={24} />
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDelete(paper.name); }}
                            className="text-slate-400 hover:text-red-500 transition-colors p-1"
                          >
                            <Trash2 size={18} />
                          </button>
                        </div>
                        <h3 className="mt-4 font-semibold truncate" title={paper.name}>{paper.name}</h3>
                        <p className="text-sm text-slate-500 mt-1">{paper.size}</p>
                      </div>
                    )) : (
                      <div className="col-span-full py-20 text-center text-slate-500 italic">
                        {paperSearch ? "No matching results found." : t.noPapers}
                      </div>
                    );
                  })()}
                </div>

                {/* Pagination Controls */}
                {(() => {
                  const filteredCount = papers.filter(p => p.name.toLowerCase().includes(paperSearch.toLowerCase())).length;
                  if (filteredCount <= papersPerPage) return null;
                  const totalPages = Math.ceil(filteredCount / papersPerPage);

                  return (
                    <div className="flex justify-center items-center gap-4 mt-8">
                      <button
                        disabled={paperPage === 1}
                        onClick={() => setPaperPage(p => p - 1)}
                        className="p-2 rounded-lg border border-slate-200 disabled:opacity-30 hover:bg-slate-100 dark:hover:bg-slate-700 transition"
                      >
                        <ChevronRight className="rotate-180" size={20} />
                      </button>
                      <span className="text-sm font-medium">
                        {lang === 'zh' ? `第 ${paperPage} / ${totalPages} 页` : `Page ${paperPage} of ${totalPages}`}
                      </span>
                      <button
                        disabled={paperPage >= totalPages}
                        onClick={() => setPaperPage(p => p + 1)}
                        className="p-2 rounded-lg border border-slate-200 disabled:opacity-30 hover:bg-slate-100 dark:hover:bg-slate-700 transition"
                      >
                        <ChevronRight size={20} />
                      </button>
                    </div>
                  );
                })()}
              </div>

              {/* Sidebar Preview in Documents Tab */}
              {previewFile && (
                <>
                  <div
                    className={`w-1 hover:w-1.5 cursor-col-resize transition-all self-stretch rounded-full ${darkMode ? 'bg-slate-700 hover:bg-blue-500' : 'bg-slate-200 hover:bg-blue-400'}`}
                    onMouseDown={startResizing}
                  />
                  <div className="flex flex-col gap-4 overflow-hidden" style={{ width: sidebarWidth }}>
                    <div className="flex items-center justify-between shrink-0">
                      <h3 className={`font-semibold flex items-center gap-2 ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                        <Search size={18} />
                        {t.preview}
                      </h3>
                      <button
                        onClick={() => setPreviewFile(null)}
                        className="text-xs text-blue-600 hover:underline"
                      >
                        {t.backToList}
                      </button>
                    </div>
                    <div className={`h-full rounded-lg border shadow-sm overflow-hidden flex flex-col transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                      <div className={`p-2 border-b text-[10px] font-mono truncate ${darkMode ? 'bg-slate-900/50 border-slate-700 text-slate-400' : 'bg-slate-50 border-slate-200 text-slate-600'}`}>
                        {previewFile.name}
                      </div>
                      <iframe
                        src={`${API_BASE}/papers/${previewFile.name}`}
                        className="w-full flex-1 border-none"
                        style={{ pointerEvents: isResizing ? 'none' : 'auto' }}
                        title={l10n.pdfViewerTitle}
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {activeTab === 'training' && (
            <div className="space-y-8 max-w-5xl">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <ActionCard title={t.indexing} desc={t.indexingDesc} icon={<RefreshCw />} action={handleIndex} onStop={handleStop} isLoading={activeTask.is_running && activeTask.task_type === 'Indexing'} isAnyRunning={activeTask.is_running || !validation.BGE_MODEL_PATH} darkMode={darkMode} stopLabel={l10n.stopLabel} executeLabel={l10n.executeLabel} />
                <ActionCard
                  title={t.sft}
                  desc={t.sftDesc}
                  icon={<Database />}
                  action={handleSFT}
                  onStop={handleStop}
                  color="bg-orange-600"
                  isLoading={activeTask.is_running && activeTask.task_type === 'SFT Generation'}
                  isAnyRunning={activeTask.is_running || (settings.SFT_MODEL_SOURCE === 'api' ? !validation.API : !validation.BASE_MODEL_ID)}
                  darkMode={darkMode}
                  stopLabel={l10n.stopLabel}
                  executeLabel={l10n.executeLabel}
                />
                <ActionCard title={t.train} desc={t.trainDesc} icon={<Play />} action={handleTrain} onStop={handleStop} color="bg-emerald-600" isLoading={activeTask.is_running && activeTask.task_type === 'Training'} isAnyRunning={activeTask.is_running || !validation.BASE_MODEL_ID} darkMode={darkMode} stopLabel={l10n.stopLabel} executeLabel={l10n.executeLabel} />
              </div>
              <div className={`rounded-xl border p-4 md:p-5 transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                <div className={`text-sm font-semibold mb-3 ${darkMode ? 'text-slate-200' : 'text-slate-700'}`}>{t.metricsTitle}</div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                  <MetricCard label={t.cpuLabel} value={formatPercent(resourceLatest?.cpu)} peak={formatPercent(resourcePeaks.cpu)} peakLabel={t.peakLabel} darkMode={darkMode} />
                  <MetricCard label={t.ramLabel} value={formatPercent(resourceLatest?.ram)} peak={formatPercent(resourcePeaks.ram)} peakLabel={t.peakLabel} darkMode={darkMode} />
                  <MetricCard label={t.gpuLabel} value={formatPercent(resourceLatest?.gpu)} peak={formatPercent(resourcePeaks.gpu)} peakLabel={t.peakLabel} darkMode={darkMode} />
                </div>
                <ResourceLineChart
                  points={resourcePoints}
                  darkMode={darkMode}
                  cpuLabel={t.cpuLabel}
                  ramLabel={t.ramLabel}
                  gpuLabel={t.gpuLabel}
                  samplesLabel={t.samplesLabel}
                  noSamplesLabel={t.noSamplesLabel}
                />
              </div>
              <div className="bg-slate-900 rounded-xl p-6 overflow-hidden flex flex-col h-[500px]">
                <div className="flex items-center gap-2 mb-4 text-slate-400 text-sm font-mono"><Terminal size={16} /><span>{l10n.trainingLogLabel}</span></div>
                <pre className="flex-1 overflow-y-auto terminal-scroll text-slate-300 font-mono text-sm whitespace-pre-wrap">{logs}</pre>
              </div>
            </div>
          )}

          {activeTab === 'chat' && (
            <div className="flex gap-6 h-full">
              <div className={`w-64 flex flex-col rounded-xl border overflow-hidden shadow-sm transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                <div className={`p-4 border-b ${darkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                  <button onClick={handleNewChat} className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition flex items-center justify-center gap-2 text-sm font-medium">
                    <Play size={16} className="rotate-90" />{t.newChat}
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                  {sessions.map((s) => (
                    <div key={s.id} onClick={() => handleSelectSession(s.id)} className={`group relative p-3 rounded-lg cursor-pointer transition-colors text-sm ${currentSessionId === s.id ? (darkMode ? 'bg-blue-900/40 text-blue-300 font-medium' : 'bg-blue-50 text-blue-600 font-medium') : (darkMode ? 'text-slate-400 hover:bg-slate-700/50' : 'text-slate-600 hover:bg-slate-50')}`}>
                      <div className="pr-6 truncate">{s.title}</div>
                      <button onClick={(e) => handleDeleteSession(e, s.id)} className="absolute right-2 top-3 opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 transition-opacity"><Trash2 size={14} /></button>
                    </div>
                  ))}
                </div>
              </div>

              <div className={`flex-1 flex flex-col rounded-xl border overflow-hidden shadow-sm transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                <div
                  ref={chatScrollRef}
                  onScroll={handleChatScroll}
                  className="flex-1 overflow-y-auto p-6 space-y-6"
                >
                  {chatHistory.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] p-4 rounded-2xl ${msg.role === 'user' ? 'bg-blue-600 text-white' : (darkMode ? 'bg-slate-700 text-slate-100' : 'bg-slate-100 text-slate-800')}`}>
                        <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'text-white prose-invert' : (darkMode ? 'prose-invert text-slate-100' : 'text-slate-800')}`}>
                          {(() => {
                            let thinking = null;
                            let answer = msg.content;

                            // Handle multiple possible think-tag formats:
                            // 1. Standard format: <think>...</think>
                            // 2. Only closing tag: ...</think> (streaming/truncated outputs)
                            if (msg.content.includes("</think>")) {
                              const parts = msg.content.split("</think>");
                              const possibleThinking = parts[0].replace("<think>", "").trim();
                              if (possibleThinking) {
                                thinking = possibleThinking;
                                answer = parts.slice(1).join("</think>").trim();
                              }
                            }

                            return (
                              <>
                                {thinking && (
                                  <details className={`mb-4 rounded-lg border text-xs ${darkMode ? 'bg-slate-800/50 border-slate-600' : 'bg-white/50 border-slate-200'}`}>
                                    <summary className="cursor-pointer p-2 font-medium text-slate-500 hover:text-blue-500 transition-colors select-none">
                                      {lang === 'zh' ? '查看思考过程' : 'View Thought Process'}
                                    </summary>
                                    <div className="p-3 pt-0 text-slate-400 italic whitespace-pre-wrap border-t border-slate-200/20 mt-1">
                                      {thinking}
                                    </div>
                                  </details>
                                )}
                                <ReactMarkdown
                                  remarkPlugins={[remarkMath]}
                                  rehypePlugins={[rehypeKatex]}
                                  components={{
                                    p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                                    code: ({ node, inline, ...props }) => inline
                                      ? <code className={`${darkMode ? 'bg-slate-600' : 'bg-slate-200'} px-1 rounded text-pink-500`} {...props} />
                                      : <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto my-2"><code {...props} /></pre>
                                  }}
                                >
                                  {answer || (thinking ? "" : msg.content)}
                                </ReactMarkdown>
                              </>
                            );
                          })()}
                        </div>
                        {msg.role === 'assistant' && Array.isArray(msg.sources) && msg.sources.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {msg.sources.map((s, idx) => (
                              <button
                                key={idx}
                                onClick={() => openEvidence(s, getNearestUserQuery(chatHistory, i))}
                                className={`text-[10px] px-2 py-0.5 rounded border transition whitespace-normal break-all text-left max-w-full ${darkMode ? 'bg-slate-800/50 hover:bg-slate-800 border-slate-600 text-slate-300' : 'bg-white/50 hover:bg-white border-slate-200 text-slate-600'}`}
                              >
                                [{s.source} p.{s.page}]
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {loading && <div className="text-slate-400 text-sm italic">{t.thinking}</div>}
                  <div ref={messagesEndRef} />
                </div>
                <div className={`p-4 border-t flex gap-4 ${darkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                  <div className="relative" ref={modeMenuRef}>
                    <button
                      onClick={() => setModeMenuOpen((v) => !v)}
                      className={`px-3 py-2 text-xs rounded-lg border whitespace-nowrap transition ${
                        compareMode || webSearchEnabled
                          ? (darkMode ? 'bg-blue-900/40 text-blue-300 border-blue-700' : 'bg-blue-50 text-blue-700 border-blue-200')
                          : (darkMode ? 'bg-slate-900/50 text-slate-300 border-slate-700' : 'bg-white text-slate-600 border-slate-200')
                      }`}
                      title={`${t.compareHint} / ${t.webSearchHint}`}
                    >
                      {t.modes}
                    </button>
                    {modeMenuOpen && (
                      <div className={`absolute bottom-full mb-2 left-0 z-20 min-w-48 rounded-lg border shadow-lg p-2 ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                        <label className={`flex items-center gap-2 px-2 py-1.5 text-xs rounded cursor-pointer ${darkMode ? 'hover:bg-slate-700 text-slate-200' : 'hover:bg-slate-50 text-slate-700'}`}>
                          <input
                            type="checkbox"
                            checked={compareMode}
                            onChange={(e) => setCompareMode(e.target.checked)}
                          />
                          <span>{t.compareMode}</span>
                        </label>
                        <label className={`flex items-center gap-2 px-2 py-1.5 text-xs rounded cursor-pointer ${darkMode ? 'hover:bg-slate-700 text-slate-200' : 'hover:bg-slate-50 text-slate-700'}`}>
                          <input
                            type="checkbox"
                            checked={webSearchEnabled}
                            onChange={(e) => setWebSearchEnabled(e.target.checked)}
                          />
                          <span>{t.webSearch}</span>
                        </label>
                      </div>
                    )}
                  </div>
                  <input
                    className={`flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors ${darkMode ? 'bg-slate-900 border-slate-700 text-slate-100' : 'bg-white border-slate-200 text-slate-900'} ${activeTask.is_running ? 'opacity-50 cursor-not-allowed' : ''}`}
                    placeholder={activeTask.is_running ? `Cannot chat while ${activeTask.task_type} is running...` : (compareMode ? `${t.askPlaceholder} (${t.compareHint})` : t.askPlaceholder)}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && !activeTask.is_running && handleChat()}
                    disabled={activeTask.is_running}
                  />
                  {loading ? (
                    <button
                      onClick={handleStopChat}
                      className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition flex items-center gap-2"
                    >
                      <div className="w-2 h-2 bg-white rounded-full animate-ping"></div>
                      {lang === 'zh' ? '停止回答' : 'Stop'}
                    </button>
                  ) : (
                    <button onClick={handleChat} disabled={loading || activeTask.is_running} className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">{t.send}</button>
                  )}
                </div>
              </div>

              <div className={`w-1 hover:w-1.5 cursor-col-resize transition-all self-stretch rounded-full ${darkMode ? 'bg-slate-700 hover:bg-blue-500' : 'bg-slate-200 hover:bg-blue-400'}`} onMouseDown={startResizing} />
              <div className="flex flex-col gap-4 overflow-hidden" style={{ width: sidebarWidth }}>
                <div className="flex items-center justify-between shrink-0">
                  <h3 className={`font-semibold flex items-center gap-2 ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}><Search size={18} />{previewFile ? t.preview : t.sources}</h3>
                  {previewFile && <button onClick={() => setPreviewFile(null)} className="text-xs text-blue-600 hover:underline">{t.backToList}</button>}
                </div>
                <div className="flex-1 overflow-y-auto space-y-4 pr-1">
                  {previewFile ? (
                    <div className="h-full flex flex-col gap-3">
                      <div className={`rounded-lg border shadow-sm overflow-hidden flex flex-col transition-colors ${selectedEvidence ? 'min-h-[58%] flex-1' : 'h-full'} ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                        <div className={`p-2 border-b text-[10px] font-mono truncate ${darkMode ? 'bg-slate-900/50 border-slate-700 text-slate-400' : 'bg-slate-50 border-slate-200 text-slate-600'}`}>{previewFile.name} (P{previewFile.page})</div>
                        <iframe src={`${API_BASE}/papers/${previewFile.name}#page=${previewFile.page}`} className="w-full flex-1 border-none" style={{ pointerEvents: isResizing ? 'none' : 'auto' }} title={l10n.pdfViewerTitle} />
                      </div>

                      {selectedEvidence && (
                        <div className={`rounded-lg border shadow-sm overflow-hidden transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                          <div className={`px-3 py-2 border-b text-xs font-semibold ${darkMode ? 'bg-slate-900/50 border-slate-700 text-slate-300' : 'bg-slate-50 border-slate-200 text-slate-700'}`}>
                            {t.evidence}
                          </div>
                          <div className="p-3">
                            <p className="text-[11px] mb-2 text-slate-500">{t.evidenceHint}</p>
                            <div className={`text-xs leading-relaxed whitespace-pre-wrap max-h-52 overflow-auto ${darkMode ? 'text-slate-200' : 'text-slate-700'}`}>
                              <HighlightedEvidence
                                text={selectedEvidence.text || ""}
                                query={selectedEvidence.query || ""}
                                darkMode={darkMode}
                              />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : debugSources.map((s, i) => (
                    <div key={i} onClick={() => openEvidence(s, getLatestUserQuery(chatHistory))} className={`p-4 rounded-lg border text-xs shadow-sm cursor-pointer transition-all ${darkMode ? 'bg-slate-800 border-slate-700 hover:border-blue-500' : 'bg-white border-slate-200 hover:border-blue-300'}`}>
                      <div className="flex justify-between items-start gap-2 mb-2 font-bold text-blue-600">
                        <span className="min-w-0 break-all leading-snug">{s.source} (P{s.page})</span>
                        <span className="shrink-0">{(s.score * 100).toFixed(1)}%</span>
                      </div>
                      <p className={`${darkMode ? 'text-slate-400' : 'text-slate-600'} leading-relaxed italic line-clamp-6`}>{s.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="max-w-4xl space-y-8 pb-10">
              <section className={`rounded-xl border overflow-hidden shadow-sm transition-colors ${darkMode ? 'bg-slate-800 border-orange-900/50' : 'bg-white border-orange-200'}`}>
                <div className={`px-6 py-3 border-b flex items-center gap-2 font-semibold ${darkMode ? 'bg-orange-900/20 border-orange-900/50 text-orange-400' : 'bg-orange-50 border-orange-200 text-orange-800'}`}><RefreshCw size={18} className="animate-spin-slow" /><h4>{t.coreConfig}</h4></div>
                <div className="p-6 space-y-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <SettingInput label={t.projectName} value={settings.PROJECT_NAME} onChange={(v) => setSettings({...settings, PROJECT_NAME: v})} darkMode={darkMode} />
                    <SettingInput label={t.backendPortLabel} value={settings.BACKEND_PORT} onChange={(v) => setSettings({...settings, BACKEND_PORT: parseInt(v) || 0})} type="number" darkMode={darkMode} />

                    <SettingSelect
                      label={t.baseLlmLabel}
                      value={settings.BASE_MODEL_ID}
                      options={availableModels.llm_hints.length ? availableModels.llm_hints : availableModels.all}
                      onChange={(v) => {
                        const nextPath = `workspace/rag_index/models/${v}`;
                        setSettings((prev) => ({ ...prev, BASE_MODEL_ID: nextPath }));
                        setValidation((prev) => ({ ...prev, BASE_MODEL_ID: null }));
                        validatePathField("BASE_MODEL_ID", nextPath);
                      }}
                      darkMode={darkMode}
                      isValid={validation.BASE_MODEL_ID}
                      placeholder={t.selectModel}
                      emptyMessage={t.noModelsFound}
                      validLabel={l10n.validLabel}
                      invalidLabel={l10n.invalidLabel}
                    />

                    <SettingSelect
                      label={t.embeddingModelLabel}
                      value={settings.BGE_MODEL_PATH}
                      options={availableModels.emb_hints.length ? availableModels.emb_hints : availableModels.all}
                      onChange={(v) => {
                        const nextPath = `workspace/rag_index/models/${v}`;
                        setSettings((prev) => ({ ...prev, BGE_MODEL_PATH: nextPath }));
                        setValidation((prev) => ({ ...prev, BGE_MODEL_PATH: null }));
                        validatePathField("BGE_MODEL_PATH", nextPath);
                      }}
                      darkMode={darkMode}
                      isValid={validation.BGE_MODEL_PATH}
                      placeholder={t.selectModel}
                      emptyMessage={t.noModelsFound}
                      validLabel={l10n.validLabel}
                      invalidLabel={l10n.invalidLabel}
                    />
                  </div>
                  <p className="text-[10px] text-orange-500 mt-2 italic">{t.restartNote}</p>
                </div>
              </section>

            <section className={`rounded-xl border overflow-hidden shadow-sm transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                <div className={`px-6 py-3 border-b flex items-center gap-2 font-semibold ${darkMode ? 'bg-slate-900/30 border-slate-700 text-slate-300' : 'bg-slate-50 border-slate-200 text-slate-800'}`}><Play size={18} /><h4>{t.runtimeConfig}</h4></div>
                <div className="p-6 space-y-1">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <SettingInput label={t.researchFieldLabel} value={settings.RESEARCH_FIELD} onChange={(v) => setSettings({...settings, RESEARCH_FIELD: v})} darkMode={darkMode} />
                    <div className="flex flex-col gap-1.5">
                      <label className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>{t.sftModelSource}</label>
                      <div className="flex gap-4 mt-1">
                        <label className="flex items-center gap-2 cursor-pointer text-sm">
                          <input type="radio" checked={settings.SFT_MODEL_SOURCE === 'local'} onChange={() => setSettings({...settings, SFT_MODEL_SOURCE: 'local'})} />
                          <span>{t.localModel}</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer text-sm">
                          <input type="radio" checked={settings.SFT_MODEL_SOURCE === 'api'} onChange={() => setSettings({...settings, SFT_MODEL_SOURCE: 'api'})} />
                          <span>{t.apiModel}</span>
                        </label>
                      </div>
                    </div>
                  </div>

                  {settings.SFT_MODEL_SOURCE === 'api' && (
                    <div className="space-y-4 border-l-2 border-blue-500/30 pl-4 mb-6 animate-in fade-in slide-in-from-left-2">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <SettingInput label={l10n.apiModelIdLabel} value={settings.API_MODEL_ID} onChange={(v) => setSettings({...settings, API_MODEL_ID: v})} darkMode={darkMode} />
                        <SettingInput label={l10n.apiBaseUrlLabel} value={settings.API_BASE_URL} onChange={(v) => setSettings({...settings, API_BASE_URL: v})} darkMode={darkMode} />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <SettingInput label={l10n.apiKeyLabel} value={settings.API_KEY} onChange={(v) => setSettings({...settings, API_KEY: v})} type="password" darkMode={darkMode} isValid={validation.API} validLabel={l10n.validLabel} invalidLabel={l10n.invalidLabel} />
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                    <SettingInput
                      label={t.chatTopKLabel}
                      value={settings.CHAT_TOPK}
                      onChange={(v) => setSettings({...settings, CHAT_TOPK: parseInt(v, 10) || 0})}
                      type="number"
                      darkMode={darkMode}
                    />
                    <SettingInput
                      label={t.compareTopKLabel}
                      value={settings.COMPARE_TOPK}
                      onChange={(v) => setSettings({...settings, COMPARE_TOPK: parseInt(v, 10) || 0})}
                      type="number"
                      darkMode={darkMode}
                    />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <SettingInput
                      label={t.chatMaxTokensLabel}
                      value={settings.CHAT_MAX_TOKENS}
                      onChange={(v) => setSettings({...settings, CHAT_MAX_TOKENS: parseInt(v, 10) || 0})}
                      type="number"
                      darkMode={darkMode}
                    />
                    <SettingInput
                      label={t.chatHistoryRoundsLabel}
                      value={settings.CHAT_HISTORY_ROUNDS}
                      onChange={(v) => setSettings({...settings, CHAT_HISTORY_ROUNDS: parseInt(v, 10) || 0})}
                      type="number"
                      darkMode={darkMode}
                    />
                  </div>
                </div>
              </section>

              <div className="flex justify-end pt-4"><button onClick={handleSaveSettings} className="px-8 py-3 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-700 shadow-lg shadow-blue-200 transition-all flex items-center gap-2"><Database size={20} />{t.saveSettings}</button></div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function HighlightedEvidence({ text, query, darkMode }) {
  const parts = splitHighlightedText(text, query);

  return (
    <>
      {parts.map((part, idx) => (
        part.highlight ? (
          <mark key={idx} className={darkMode ? 'bg-yellow-400/40 text-yellow-100 px-0.5 rounded' : 'bg-yellow-200 text-slate-900 px-0.5 rounded'}>{part.text}</mark>
        ) : (
          <span key={idx}>{part.text}</span>
        )
      ))}
    </>
  );
}

function formatPercent(value) {
  if (value == null) return 'N/A';
  return `${value.toFixed(1)}%`;
}

function MetricCard({ label, value, peak, peakLabel, darkMode }) {
  return (
    <div className={`rounded-lg border p-3 transition-colors ${darkMode ? 'bg-slate-900/40 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
      <div className={`text-[11px] uppercase tracking-wide ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>{label}</div>
      <div className={`text-lg font-semibold mt-1 ${darkMode ? 'text-slate-100' : 'text-slate-900'}`}>{value}</div>
      <div className={`text-[11px] mt-1 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>{peakLabel}: {peak}</div>
    </div>
  );
}

function ResourceLineChart({ points, darkMode, cpuLabel, ramLabel, gpuLabel, samplesLabel, noSamplesLabel }) {
  const width = 820;
  const height = 180;
  const pad = 14;
  const values = points
    .flatMap((p) => [p.cpu, p.ram, p.gpu])
    .filter((v) => v != null);
  const maxValue = Math.max(100, values.length ? Math.max(...values) : 100);
  const series = points.length ? points : [];

  const toPath = (key) => {
    if (!series.length) return '';
    let d = '';
    for (let i = 0; i < series.length; i += 1) {
      const value = series[i][key];
      if (value == null) continue;
      const x = pad + (i / Math.max(1, series.length - 1)) * (width - pad * 2);
      const y = height - pad - (value / maxValue) * (height - pad * 2);
      d += `${d ? ' L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    return d;
  };

  const cpuPath = toPath('cpu');
  const ramPath = toPath('ram');
  const gpuPath = toPath('gpu');

  return (
    <div className={`rounded-lg border p-3 transition-colors ${darkMode ? 'bg-slate-900/30 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
      <div className="flex flex-wrap items-center gap-3 text-xs mb-2">
        <LegendDot color="#22c55e" label={cpuLabel} />
        <LegendDot color="#38bdf8" label={ramLabel} />
        <LegendDot color="#f59e0b" label={gpuLabel} />
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-44">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke={darkMode ? '#475569' : '#cbd5e1'} strokeWidth="1" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke={darkMode ? '#475569' : '#cbd5e1'} strokeWidth="1" />
        {cpuPath && <path d={cpuPath} fill="none" stroke="#22c55e" strokeWidth="2.2" strokeLinecap="round" />}
        {ramPath && <path d={ramPath} fill="none" stroke="#38bdf8" strokeWidth="2.2" strokeLinecap="round" />}
        {gpuPath && <path d={gpuPath} fill="none" stroke="#f59e0b" strokeWidth="2.2" strokeLinecap="round" />}
      </svg>
      <div className={`text-[11px] mt-2 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>
        {series.length ? samplesLabel.replace('{count}', String(series.length)) : noSamplesLabel}
      </div>
    </div>
  );
}

function LegendDot({ color, label }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      <span>{label}</span>
    </span>
  );
}

function NavItem({ icon, label, active, onClick, darkMode }) {
  return (
    <button onClick={onClick} className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${active ? (darkMode ? 'bg-blue-900/40 text-blue-300 font-medium' : 'bg-blue-50 text-blue-600 font-medium') : (darkMode ? 'text-slate-400 hover:bg-slate-700/50' : 'text-slate-600 hover:bg-slate-50')}`}>
      {React.cloneElement(icon, { size: 20 })}<span className="truncate">{label}</span>{active && <ChevronRight className="ml-auto shrink-0" size={16} />}
    </button>
  );
}

function ActionCard({ title, desc, icon, action, onStop, isLoading, isAnyRunning, color = "bg-blue-600", darkMode, stopLabel = "Stop", executeLabel = "Execute" }) {
  return (
    <div className={`p-6 rounded-xl border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm overflow-hidden transition-colors ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
      <div className="flex items-center gap-4 flex-1 min-w-0">
        <div className={`p-3 rounded-lg text-white shrink-0 ${color}`}>{React.cloneElement(icon, { size: 24 })}</div>
        <div className="min-w-0"><h4 className={`font-semibold truncate ${darkMode ? 'text-slate-100' : 'text-slate-900'}`}>{title}</h4><p className="text-sm text-slate-500 break-words line-clamp-2" title={desc}>{desc}</p></div>
      </div>
      <div className="flex gap-2 w-full sm:w-auto shrink-0 justify-end">
        {isLoading ? <button onClick={onStop} className={`flex-1 sm:flex-none px-4 py-2 border rounded-lg transition flex items-center justify-center gap-2 whitespace-nowrap ${darkMode ? 'bg-red-900/20 text-red-400 border-red-900/50 hover:bg-red-900/30' : 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100'}`}><div className="w-2 h-2 bg-red-600 rounded-full animate-ping"></div>{stopLabel}</button> : <button onClick={action} disabled={isAnyRunning} className={`flex-1 sm:flex-none px-4 py-2 border rounded-lg transition whitespace-nowrap ${darkMode ? 'bg-slate-900/50 border-slate-700 text-slate-400 hover:bg-slate-700 disabled:opacity-30' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50'}`}>{executeLabel}</button>}
      </div>
    </div>
  );
}

function SettingSelect({ label, value, options, onChange, darkMode, isValid = null, placeholder = "Select...", emptyMessage = "No models found", validLabel = "VALID", invalidLabel = "INVALID" }) {
  // 从完整路径中提取目录名，用于匹配下拉框选项
  const currentValue = value ? value.split('/').pop() : "";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-center">
        <label className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>{label}</label>
        {isValid !== null && (
          <span className={`text-[10px] px-1.5 rounded-full font-bold ${isValid ? 'bg-emerald-500/20 text-emerald-500' : 'bg-red-500/20 text-red-500'}`}>
            {isValid ? validLabel : invalidLabel}
          </span>
        )}
      </div>
      <select
        value={currentValue}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition appearance-none cursor-pointer ${darkMode ? 'bg-slate-900 border-slate-700 text-slate-100' : 'bg-white border-slate-200 text-slate-900'}`}
      >
        <option value="" disabled>{placeholder}</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
        {options.length === 0 && <option value="" disabled>{emptyMessage}</option>}
      </select>
    </div>
  );
}

function SettingInput({ label, value, onChange, type = "text", darkMode, isValid = null, validLabel = "VALID", invalidLabel = "INVALID" }) {
  const [showPassword, setShowPassword] = useState(false);
  const isPassword = type === "password";
  const inputType = isPassword ? (showPassword ? "text" : "password") : type;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-center">
        <label className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>{label}</label>
        {isValid !== null && (
          <span className={`text-[10px] px-1.5 rounded-full font-bold ${isValid ? 'bg-emerald-500/20 text-emerald-500' : 'bg-red-500/20 text-red-500'}`}>
            {isValid ? validLabel : invalidLabel}
          </span>
        )}
      </div>
      <div className="relative">
        <input type={inputType} value={value || ""} onChange={(e) => onChange(e.target.value)} className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition pr-10 ${darkMode ? 'bg-slate-900 border-slate-700 text-slate-100' : 'bg-white border-slate-200 text-slate-900'}`} />
        {isPassword && <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors">{showPassword ? <EyeOff size={16} /> : <Eye size={16} />}</button>}
      </div>
    </div>
  );
}

