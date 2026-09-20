# ============================================================
# data/sample_data.py - 独立的数据填充脚本
# 运行方式：python data/sample_data.py
# 功能：初始化MySQL表 + 填充样例数据 + 初始化Qdrant向量库
# 可以在启动应用前单独运行此脚本完成数据准备
# ============================================================

import sys
import os

# 将项目根目录加入Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import init_database, seed_sample_data
from backend.qdrant_client import init_qdrant, seed_qdrant_samples


def main():
    print("\n" + "=" * 60)
    print("  舆情危机预警系统 - 数据初始化脚本")
    print("=" * 60 + "\n")

    # 1. MySQL初始化
    print("[步骤1] 初始化MySQL数据库...")
    try:
        init_database()
        seed_sample_data()
        print("  ✓ MySQL初始化完成\n")
    except Exception as e:
        print(f"  ✗ MySQL初始化失败: {e}")
        print("  请确保MySQL已启动，且config中的用户名/密码正确\n")

    # 2. Qdrant初始化
    print("[步骤2] 初始化Qdrant向量库...")
    try:
        init_qdrant()
        seed_qdrant_samples()
        print("  ✓ Qdrant初始化完成\n")
    except Exception as e:
        print(f"  ✗ Qdrant初始化失败: {e}")
        print("  请确保Qdrant已启动（docker run -p 6333:6333 qdrant/qdrant）\n")

    print("=" * 60)
    print("  数据初始化完成！可以启动应用：python start.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
