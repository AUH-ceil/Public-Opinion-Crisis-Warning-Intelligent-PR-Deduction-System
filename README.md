# 舆情危机预警与智能公关推演系统

基于 **LangGraph 多智能体** 的跨平台舆情危机预警与智能公关推演系统。输入一个品牌名称，系统自动完成舆情抓取、情感分析、RAG 历史案例检索、竞品反应推演、风险指数计算与分级公关策略生成，帮助品牌在危机发酵前提前预警、快速响应。

## ✨ 核心功能

- **舆情爬虫**：模拟微博 / 抖音 / 小红书 / 新闻媒体等多平台舆情抓取，支持真实搜索引擎三级降级
- **LangGraph 多智能体**：爬虫 → 情感分析 → 竞品推演 → 公关策略，4 大 Agent 协同分析
- **RAG 历史案例**：Qdrant 向量库 + TF-IDF 检索相似历史危机案例
- **危机推演**：热度走势预测（2h / 12h / 24h）+ 上热搜概率
- **四级预警**：低 / 中 / 高 / 特级，自动弹窗告警
- **公关方案**：紧急 / 常规 / 长期三层应对策略，含话术要点与建议声明
- **报告导出**：一键生成完整公关处理报告

## 🛠 技术栈

| 分类 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| 多智能体 | LangGraph（StateGraph 7 节点 DAG） |
| 大模型 | DeepSeek（OpenAI 兼容 API） |
| 向量检索 | Qdrant + sklearn TF-IDF |
| 数据库 | MySQL（pymysql） |
| 搜索引擎 | SerpAPI → Playwright（多级降级） |
| 数据校验 | Pydantic |
| 前端 | 原生 HTML/CSS/JS + Chart.js |

## 🏗 架构

```
[爬虫] → [情感分析] ──┬── [RAG检索]   ──┐
                      │       (并行)     ├→ [竞品反应] → [风险计算] → [公关策略]
                      └── [MySQL查询] ──┘
```

工作流为 7 节点 DAG，其中「RAG 检索」与「MySQL 查询」并行执行。每个 Agent 都设计了降级链：LLM 可用时调用 DeepSeek 推理，不可用时自动切换规则引擎 / 模板匹配，保证任意环境下系统都能产出结果。

**风险指数公式**：

```
风险指数 = min(100, 负面情感强度 × 媒体权重 × 传播热度 × 100)
  负面情感强度 = Σ|负面情感分| / 总条数
  传播热度     = log10(触达人数 + 1) / 10
```

## 📁 目录结构

```
agent11/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── database.py          # MySQL 数据操作
│   ├── llm_client.py        # LLM 统一调用封装（指数退避重试 + 降级）
│   ├── search.py            # 搜索引擎（SerpAPI → Google CSE → Playwright）
│   ├── qdrant_client.py     # 向量库（Qdrant 不可用时降级为本地检索）
│   ├── models.py            # Pydantic 数据模型
│   ├── risk_calculator.py   # 风险指数计算 + 走势预测
│   ├── api/
│   │   └── routes.py        # API 路由
│   ├── agents/
│   │   ├── crawler_agent.py       # 爬虫 Agent
│   │   ├── sentiment_agent.py     # 情感分析 Agent
│   │   ├── competitor_agent.py    # 竞品推演 Agent
│   │   ├── pr_strategy_agent.py   # 公关策略 Agent
│   │   └── pr_templates.py        # 策略模板
│   └── workflow/
│       ├── graph.py         # LangGraph 工作流构建 + 执行入口
│       └── nodes.py         # 各节点实现
├── data/
│   ├── init_db.sql          # 数据库初始化 SQL
│   └── sample_data.py       # 样例数据
├── frontend/
│   ├── index.html           # 前端页面
│   ├── css/style.css
│   └── js/app.js
├── start.py                 # 一键启动脚本
└── requirements.txt
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- MySQL 8.0+
- （可选）Qdrant 向量库，不启动时自动降级为本地内存检索

### 2. 安装依赖

```bash
pip install -r requirements.txt

# 可选：安装浏览器内核（用于 Playwright 搜索降级）
playwright install chromium
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，填入以下配置（`.env` 已被 `.gitignore` 排除，不会上传）：

```bash
# 必填：DeepSeek API Key（https://platform.deepseek.com/api_keys 获取）
DEEPSEEK_API_KEY=sk-xxxx

# 可选：模型与 API 地址
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1

# 可选：搜索引擎
SERPAPI_KEY=
# GOOGLE_API_KEY=
# GOOGLE_CX=

# 数据库
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=crisis_db

# 向量库
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 4. 启动

```bash
python start.py
```

启动后访问：

| 地址 | 说明 |
|---|---|
| http://localhost:8000/app | 前端页面 |
| http://localhost:8000/docs | API 文档（Swagger） |

首次启动会自动初始化数据库（建表 + 样例数据）与向量库。

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 系统首页 |
| POST | `/api/analyze` | 一键舆情危机分析（核心接口） |
| GET | `/api/history` | 历史舆情记录 |
| GET | `/api/media-weights` | 媒体权重数据 |
| GET | `/api/stats` | 系统统计（风险分布等） |
| POST | `/api/export-report` | 导出公关处理报告 |
| GET | `/api/health` | 健康检查（含 LLM 状态） |

**核心分析示例**：

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"brand_name": "某奶茶品牌", "query": "食品安全", "simulate_count": 20}'
```

## 📝 说明

- **多级降级**：SerpAPI 搜索不可用 → 自动切换 Google CSE → Playwright 浏览器搜索 → 基于品类动态生成兜底数据；LLM 不可用 → 自动切换规则引擎。
- **离线可用**：Qdrant / MySQL 未启动时，系统仍能以本地内存模式运行，只是 RAG 与数据持久化功能受限。
- 本项目为演示 / 学习用途，风险计算模型为简化公式，不构成真实舆情研判依据。
