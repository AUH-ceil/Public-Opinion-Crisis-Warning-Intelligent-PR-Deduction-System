# ============================================================
# backend/main.py - FastAPI主入口（精简版）
# 路由逻辑在 api/routes.py | 工作流在 workflow/
# 启动: python start.py 或 uvicorn backend.main:app
# ============================================================

# === 最先加载 .env（必须在其他模块 import 之前） ===
from dotenv import load_dotenv
import os

# 查找项目根目录的 .env 文件
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_env_path)
print(f"[配置] 已加载环境变量: {_env_path}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.database import init_database, seed_sample_data
from backend.qdrant_client import init_qdrant, seed_qdrant_samples


# ==================== 应用初始化 ====================

app = FastAPI(
    title="舆情危机预警与智能公关推演系统",
    description="""
    ## 跨平台舆情危机预警与智能公关推演系统

    ### 核心功能：
    - **舆情爬虫**：模拟全网平台文本抓取
    - **LangGraph多智能体**：4大Agent并行分析（StateGraph全局状态管理）
    - **RAG历史案例**：Qdrant向量库相似案例检索
    - **危机推演**：热度走势预测 + 热搜概率
    - **预警分级**：低/中/高/特级四级预警 + 自动弹窗
    - **公关方案**：紧急/常规/长期三分层应对策略
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup():
    """启动时自动初始化数据库 + 向量库"""
    print("\n" + "=" * 60)
    print("  舆情危机预警与智能公关推演系统 v1.0.0")
    print("  API文档: http://localhost:8000/docs")
    print("  前端页面: http://localhost:8000/app")
    print("=" * 60 + "\n")

    try:
        init_database()
        seed_sample_data()
    except Exception as e:
        print(f"[警告] 数据库初始化失败: {e}")
        print("[提示] 系统将在无数据库模式下运行")

    try:
        init_qdrant()
        seed_qdrant_samples()
    except Exception as e:
        print(f"[警告] Qdrant初始化失败: {e}")
        print("[提示] RAG检索功能将不可用")


# ==================== 前端静态文件 ====================

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(FRONTEND_DIR):
    @app.get("/app", tags=["前端"])
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/app/", tags=["前端"])
    async def serve_frontend_slash():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # 挂载静态资源目录（CSS/JS文件）
    css_dir = os.path.join(FRONTEND_DIR, "css")
    js_dir = os.path.join(FRONTEND_DIR, "js")
    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")
