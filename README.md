# Aegis - 多智能体个人工作助理

基于 **CrewAI** 构建的多智能体系统，作为个人工作助理，
支持文件深度处理、任务自动分派、上下文对话记忆、智能提醒和研究分析。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        👤 用户界面                           │
│                 (CLI / API / 聊天界面)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ 用户请求
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     🎯 Orchestrator (调度器)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ 意图识别  │  │ 工作流编排│  │ 消息路由  │  │ 结果聚合   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
└───────┬──────────────┬──────────────┬──────────────┬────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 文件处理  │  │ 任务分派  │  │ 对话记忆  │  │ 智能提醒  │  │ 研究分析  │
│  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │             │
     └─────────────┴──────┬──────┴─────────────┴─────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐  ┌────────────┐  ┌──────────────┐
    │ LLM 接口  │  │ 记忆系统    │  │ 工具集        │
    │ (OpenAI) │  │ (短期+长期) │  │ (文件/搜索/..) │
    └──────────┘  └────────────┘  └──────────────┘
```

## 项目结构

```
Aegis/
├── main.py                          # 主入口 (交互模式 + CLI)
├── config.yaml                      # 全局配置
├── .env.example                     # 环境变量模板
├── requirements.txt                 # Python 依赖
├── src/
│   ├── core/
│   │   ├── agents/                  # 🤖 智能体定义
│   │   │   ├── base.py              #   基类 (CrewAI 封装)
│   │   │   ├── file_processor.py    #   文件处理代理
│   │   │   ├── task_dispatcher.py   #   任务分派代理
│   │   │   ├── memory_agent.py      #   对话记忆代理
│   │   │   ├── reminder_agent.py    #   智能提醒代理
│   │   │   ├── research_agent.py    #   研究分析代理
│   │   │   └── orchestrator.py      #   调度器 (含 MessageBus)
│   │   ├── tools/                   # 🔧 工具集
│   │   │   ├── file_tools.py        #   文件解析/生成
│   │   │   ├── search_tools.py      #   搜索工具
│   │   │   ├── translation_tools.py #   翻译/润色
│   │   │   ├── calendar_tools.py    #   日历/提醒
│   │   │   └── mcp_tools.py         #   MCP 外部连接
│   │   ├── memory/                  # 🧠 记忆系统
│   │   │   ├── short_term.py        #   短期记忆 (滑动窗口)
│   │   │   ├── long_term.py         #   长期记忆 (ChromaDB)
│   │   │   └── retriever.py         #   混合检索器
│   │   └── llm/                     # 🔌 LLM 封装
│   │       └── provider.py          #   OpenAI 兼容接口
│   ├── workflows/                   # 📋 预定义工作流
│   │   ├── file_processing.py       #   文件处理流程
│   │   ├── research_report.py       #   研究分析流程
│   │   └── reminder_followup.py     #   提醒跟进流程
│   ├── models/
│   │   └── schemas.py               #   所有数据模型 (Pydantic)
│   └── utils/
│       ├── config.py                #   配置加载器
│       └── logger.py                #   日志模块
├── tests/
│   └── test_agents.py               # 单元测试 + 集成测试
└── data/                            # 运行时数据 (自动创建)
    ├── chroma_db/                   # ChromaDB 持久化
    └── reminders.json               # 提醒持久化
```

## 快速开始

### 1. 安装依赖
```bash
cd d:/Aegis
pip install -r requirements.txt
```

### 2. 配置环境
```bash
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY
```

### 3. 运行
```bash
# 运行完整示例 (模拟会议纪要处理流程)
python main.py --example

# 交互模式
python main.py

# 处理真实文件
python main.py --file ./docs/meeting.docx --action translate

# 研究分析
python main.py --research "2024年AI发展趋势" --output report.md

# 检查提醒
python main.py --check-reminders
```

### 4. 测试
```bash
python -m pytest tests/test_agents.py -v
```

## 核心设计决策

| 决策 | 理由 |
|------|------|
| **CrewAI 多 Agent 协作** | CrewAI 提供结构化任务编排和灵活的多 Agent 对话 |
| **LLM 统一封装** | 通过 provider.py 抽象，一键切换 OpenAI/Anthropic/Azure/本地模型 |
| **记忆双层设计** | 短期记忆 (会话连贯) + 长期记忆 (跨会话知识) |
| **Pydantic 数据模型** | 所有跨 Agent 数据使用类型约束，避免 dict 传递错误 |
| **MCP 可插拔架构** | 外部工具通过 MCP 协议连接，支持热插拔和优雅降级 |
| **消息总线** | 发布-订阅模式实现 Agent 间松耦合通信 |

## 智能体协作流程

### 文件处理流程
```
User → Orchestrator → FileProcessor (解析+翻译)
                    → TaskDispatcher (提取待办+分派)
                    → ReminderAgent (设置提醒)
                    → MemoryAgent (存储关键信息)
                    → User (返回结果)
```

### 研究分析流程
```
User → Orchestrator → MemoryAgent (检索历史研究)
                    → ResearchAgent (搜索+分析+报告)
                    → MemoryAgent (存储报告摘要)
                    → User (返回报告)
```

## 可扩展点

- **新文件格式**: 在 `file_tools.py` 的 `FILE_PARSERS` 注册新解析器
- **新 LLM 提供商**: 在 `llm/provider.py` 添加新的工厂方法
- **新 Agent**: 继承 `BaseAgent`，定义 role/goal/backstory 和 execute()
- **新工作流**: 在 `workflows/` 添加新的编排函数
- **新 MCP 服务**: 在 `config.yaml` 的 `mcp_servers` 配置新端点
- **新通知方式**: 在 `calendar_tools.py` 注册新的 notify_handler

## License

MIT
