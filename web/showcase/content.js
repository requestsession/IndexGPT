const showcaseSite = {
  githubUrl: 'https://github.com/requestsession/IndexGPT',
  demoUrl: '#screen-list',
  githubText: {
    zh: '下载项目代码',
    en: 'Get the Code'
  },
  demoText: {
    zh: '查看页面截图',
    en: 'View Screenshots'
  },
  zh: {
    navMark: 'IndexGPT',
    badge: '开源科研智能体平台',
    kicker: 'Research-native LLM Workflow',
    title: '让论文检索、证据问答与模型微调\n在一个工作流里完成',
    description: '上传论文就能开始问答，回答可回溯到原文证据；从文献管理到模型训练，一套流程即可跑通。',
    sectionTitle: '核心能力',
    screenTitle: '前台页面预览',
    bilingualTitle: '项目简介（中 / EN）',
    zhLabel: '中文',
    enLabel: 'English',
    tags: ['Evidence First', 'Open Source', 'Production Ready']
  },
  en: {
    navMark: 'IndexGPT',
    badge: 'Open-source research AI workflow',
    kicker: 'Research-native LLM Workflow',
    title: 'Paper retrieval, evidence-grounded QA,\nand model tuning in one stack',
    description: 'Upload papers and ask right away. Every answer links back to evidence, while one workflow covers document ops through model training.',
    sectionTitle: 'Core Capabilities',
    screenTitle: 'Frontend Screenshots',
    bilingualTitle: 'Project Intro (ZH / EN)',
    zhLabel: 'Chinese',
    enLabel: 'English',
    tags: ['Evidence First', 'Open Source', 'Production Ready']
  },
  projectIntro: {
    zh: 'IndexGPT 面向科研团队打造，整合 PDF 解析、RAG 检索、跨论文对比、SFT 数据生成和 LoRA 训练。你可以直接部署、快速扩展，并将回答与证据严格对齐。',
    en: 'IndexGPT is built for research teams and unifies PDF ingestion, RAG retrieval, cross-paper comparison, SFT data generation, and LoRA fine-tuning. Deploy quickly, extend safely, and keep answers grounded in sources.'
  },
  metrics: {
    zh: [
      { value: 'RAG + SFT + LoRA', label: '统一流水线' },
      { value: 'Evidence First', label: '答案可追溯' },
      { value: 'OpenAI-Compatible', label: '接口易集成' }
    ],
    en: [
      { value: 'RAG + SFT + LoRA', label: 'Unified Pipeline' },
      { value: 'Evidence First', label: 'Traceable Answers' },
      { value: 'OpenAI-Compatible', label: 'Easy Integration' }
    ]
  },
  features: {
    zh: [
      '文献问答与证据定位：直接回溯原始 PDF 片段，降低幻觉风险。',
      '跨论文对比分析：在同一对话中进行多文献证据整合与观点归纳。',
      '训练闭环能力：索引构建、SFT 数据生成、LoRA 训练一站式执行。'
    ],
    en: [
      'Evidence-grounded QA: jump from answers to exact PDF snippets for verification.',
      'Cross-paper comparison: synthesize findings across multiple papers in one workflow.',
      'Training-ready loop: run indexing, SFT generation, and LoRA tuning end to end.'
    ]
  },
  screenshots: {
    zh: [
      {
        title: 'AI 助手',
        description: '流式对话、证据片段联动与会话管理。',
        image: './figs/1_AI_Assistant.png'
      },
      {
        title: '文献库',
        description: 'PDF 文献上传、检索与预览管理。',
        image: './figs/2_Documents.png'
      },
      {
        title: '训练实验室',
        description: '索引构建、SFT 生成与 LoRA 训练控制。',
        image: './figs/3_Training_Lab.png'
      },
      {
        title: '系统设置',
        description: '模型路径、API 参数与运行时配置。',
        image: './figs/4_System_Settings.png'
      }
    ],
    en: [
      {
        title: 'AI Assistant',
        description: 'Streaming chat, evidence linking, and session management.',
        image: './figs/1_AI_Assistant.png'
      },
      {
        title: 'Documents',
        description: 'PDF upload, search, and preview management.',
        image: './figs/2_Documents.png'
      },
      {
        title: 'Training Lab',
        description: 'Control indexing, SFT generation, and LoRA training.',
        image: './figs/3_Training_Lab.png'
      },
      {
        title: 'System Settings',
        description: 'Model paths, API parameters, and runtime settings.',
        image: './figs/4_System_Settings.png'
      }
    ]
  }
};

if (typeof window !== 'undefined') {
  window.showcaseSite = showcaseSite;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { showcaseSite };
}

