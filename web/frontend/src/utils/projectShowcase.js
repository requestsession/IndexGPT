export const showcaseContent = {
  githubUrl: 'https://github.com/your-org/indexgpt',
  docsUrl: 'https://github.com/your-org/indexgpt#readme',
  zh: {
    badge: '面向时序图研究的开源 AI 助手',
    title: 'IndexGPT：让论文检索、对比与问答更快落地',
    subtitle: '一个聚焦科研场景的开源项目，集成 PDF 解析、RAG 检索、跨论文对比、SFT 数据生成与 LoRA 训练流程。',
    features: [
      {
        title: '论文级问答与证据追溯',
        desc: '支持从原始 PDF 片段回溯证据，帮助你快速验证回答可信度。'
      },
      {
        title: '训练流程一体化',
        desc: '内置索引构建、SFT 数据生成与 LoRA 训练入口，减少实验切换成本。'
      },
      {
        title: '面向团队的可扩展架构',
        desc: '前后端解耦，支持 OpenAI 兼容接口，便于接入自定义模型与私有部署。'
      }
    ],
    ctaPrimary: '访问 GitHub 仓库',
    ctaSecondary: '查看项目文档'
  },
  en: {
    badge: 'Open-source AI assistant for temporal graph research',
    title: 'IndexGPT: Faster paper retrieval, comparison, and QA',
    subtitle: 'A research-oriented open-source project that combines PDF parsing, RAG retrieval, cross-paper comparison, SFT data generation, and LoRA fine-tuning workflows.',
    features: [
      {
        title: 'Paper-grounded answers with evidence traceability',
        desc: 'Jump from model output back to exact PDF snippets to validate response quality quickly.'
      },
      {
        title: 'Integrated model-training workflow',
        desc: 'Run indexing, SFT dataset generation, and LoRA training in one unified workspace.'
      },
      {
        title: 'Built for extensibility and team collaboration',
        desc: 'Decoupled frontend/backend architecture with OpenAI-compatible APIs for custom models and private deployment.'
      }
    ],
    ctaPrimary: 'Open GitHub Repository',
    ctaSecondary: 'Read the Documentation'
  }
};

