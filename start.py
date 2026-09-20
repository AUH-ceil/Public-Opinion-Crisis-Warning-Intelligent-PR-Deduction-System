# ============================================================
# start.py - 系统一键启动脚本
# 使用方式：python start.py
# 默认监听 http://localhost:8000
# 前端页面：http://localhost:8000/app
# API文档：http://localhost:8000/docs
# ============================================================

import uvicorn
import sys
import os

# === 最先加载 .env 文件 ===
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  舆情危机预警与智能公关推演系统 v1.0.0")
    print("  启动中...")
    print("=" * 60)
    print()
    print("  前端页面：http://localhost:8000/app")
    print("  API文档：http://localhost:8000/docs")
    print()
    print("  首次运行？请先执行数据初始化：")
    print("  python data/sample_data.py")
    print()
    print("  退出：Ctrl + C")
    print("=" * 60 + "\n")

    # FastAPI应用入口：backend.main:app
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # 热重载（代码修改后自动重启）
        log_level="info",
    )
