# ============================================================
# backend/database.py - MySQL数据库操作（轻量ORM风格）
# 两张核心表：media_weights（媒体权重）、history_records（历史舆情）
# 使用pymysql直连，SQL极简，无复杂封装
# ============================================================

import pymysql
import os
from typing import List, Dict, Any, Optional
from datetime import date


# ==================== 数据库配置（从 .env 读取） ====================
DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "crisis_db"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


def get_connection():
    """获取数据库连接（每次调用新建连接，简单粗暴）"""
    return pymysql.connect(**DB_CONFIG)


# ==================== 初始化建表 ====================

def init_database():
    """
    初始化数据库：建库 + 建表 + 插入样例数据
    如果库/表已存在则跳过
    """
    # 先建库（不指定database连MySQL）
    config_no_db = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    config_no_db.pop("cursorclass", None)
    conn = pymysql.connect(**config_no_db)
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS crisis_db DEFAULT CHARSET utf8mb4")
    conn.commit()
    cur.close()
    conn.close()

    conn = get_connection()
    cur = conn.cursor()

    # === 表1：媒体权重表 ===
    cur.execute("""
        CREATE TABLE IF NOT EXISTS media_weights (
            id INT AUTO_INCREMENT PRIMARY KEY,
            platform VARCHAR(32) NOT NULL COMMENT '平台名称',
            account_level VARCHAR(16) DEFAULT '腰部' COMMENT '账号级别：头部/腰部/尾部',
            weight FLOAT DEFAULT 1.0 COMMENT '权重系数',
            avg_reach INT DEFAULT 0 COMMENT '平均触达人数',
            category VARCHAR(64) DEFAULT '综合' COMMENT '领域分类'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='媒体权重表'
    """)

    # === 表2：历史结构化舆情数据表 ===
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            event_name VARCHAR(128) NOT NULL COMMENT '事件名称',
            category VARCHAR(64) DEFAULT '产品质量' COMMENT '事件分类',
            platform VARCHAR(32) DEFAULT '微博' COMMENT '首发平台',
            negative_score FLOAT DEFAULT 0 COMMENT '负面分值',
            media_weight FLOAT DEFAULT 1.0 COMMENT '媒体权重',
            risk_index FLOAT DEFAULT 0 COMMENT '风险指数',
            hot_search_hours INT DEFAULT 0 COMMENT '上热搜耗时(小时)',
            resolution TEXT COMMENT '处理方案',
            occurred_at DATE COMMENT '发生日期'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史舆情数据表'
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("[数据库] 初始化完成：crisis_db 数据库 + 两张核心表已就绪")


# ==================== 媒体权重操作 ====================

def insert_media_weight(platform: str, account_level: str, weight: float,
                        avg_reach: int, category: str = "综合"):
    """插入一条媒体权重记录"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO media_weights (platform, account_level, weight, avg_reach, category) "
        "VALUES (%s, %s, %s, %s, %s)",
        (platform, account_level, weight, avg_reach, category)
    )
    conn.commit()
    cur.close()
    conn.close()


def get_all_media_weights() -> List[Dict[str, Any]]:
    """查询所有媒体权重数据"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM media_weights")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_platform_weight(platform: str) -> float:
    """
    获取某平台的平均媒体权重
    用于风险指数计算: 媒体权重 × 负面情感分 × 传播热度
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT AVG(weight) as avg_w, SUM(avg_reach) as total_reach "
        "FROM media_weights WHERE platform = %s",
        (platform,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row and row["avg_w"]:
        return round(row["avg_w"], 2)
    return 1.0


def get_total_audience_reach() -> int:
    """计算所有媒体总触达人数"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT SUM(avg_reach) as total FROM media_weights")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["total"] if row and row["total"] else 0


# ==================== 历史记录操作 ====================

def insert_history_record(event_name: str, category: str, platform: str,
                          negative_score: float, media_weight: float,
                          risk_index: float, hot_search_hours: int,
                          resolution: str, occurred_at: date):
    """插入一条历史舆情记录"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history_records (event_name, category, platform, negative_score, "
        "media_weight, risk_index, hot_search_hours, resolution, occurred_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (event_name, category, platform, negative_score, media_weight,
         risk_index, hot_search_hours, resolution, occurred_at)
    )
    conn.commit()
    cur.close()
    conn.close()


def get_all_history_records() -> List[Dict[str, Any]]:
    """查询所有历史舆情记录"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM history_records ORDER BY occurred_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_history_by_category(category: str) -> List[Dict[str, Any]]:
    """按分类查询历史舆情"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM history_records WHERE category = %s ORDER BY risk_index DESC",
        (category,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def seed_sample_data():
    """
    插入样例数据（仅当表为空时执行）
    包含主流平台媒体权重 + 经典舆情危机案例
    """
    conn = get_connection()
    cur = conn.cursor()

    # 检查是否已有数据
    cur.execute("SELECT COUNT(*) as cnt FROM media_weights")
    if cur.fetchone()["cnt"] > 0:
        cur.close()
        conn.close()
        print("[数据库] 样例数据已存在，跳过填充")
        return

    # === 媒体权重样例数据 ===
    media_data = [
        ("微博", "头部", 2.5, 5000000, "综合"),
        ("微博", "腰部", 1.5, 500000, "综合"),
        ("微博", "尾部", 0.8, 10000, "综合"),
        ("抖音", "头部", 2.8, 8000000, "综合"),
        ("抖音", "腰部", 1.6, 800000, "综合"),
        ("抖音", "尾部", 0.9, 20000, "综合"),
        ("小红书", "头部", 2.0, 2000000, "美妆"),
        ("小红书", "腰部", 1.3, 300000, "美妆"),
        ("新闻媒体", "头部", 3.0, 10000000, "综合"),
        ("新闻媒体", "腰部", 2.0, 2000000, "综合"),
        ("知乎", "头部", 1.8, 1500000, "科技"),
        ("贴吧", "腰部", 1.2, 200000, "综合"),
    ]
    for row in media_data:
        cur.execute(
            "INSERT INTO media_weights (platform, account_level, weight, avg_reach, category) "
            "VALUES (%s, %s, %s, %s, %s)", row
        )

    # === 历史舆情案例数据 ===
    history_data = [
        ("某奶茶品牌食品安全事件", "食品安全", "微博", 0.85, 2.8, 78.5, 4,
         "立即下架问题产品，公开道歉，邀请第三方检测，赔偿消费者，CEO直播致歉", "2024-03-15"),
        ("某手机电池爆炸事件", "产品质量", "新闻媒体", 0.92, 3.0, 88.2, 2,
         "全球召回问题批次，成立专项调查组，公布调查结果，升级品控流程，赔偿用户", "2024-01-20"),
        ("某酒店服务纠纷事件", "服务纠纷", "抖音", 0.72, 2.3, 62.0, 8,
         "公开道歉信，免去涉事人员职务，全面服务培训，补偿住客，接受社会监督", "2024-05-10"),
        ("某电商平台售假事件", "产品质量", "微博", 0.88, 2.6, 80.5, 3,
         "封禁涉事店铺，设立假货举报基金，引入区块链溯源，假一赔十承诺", "2024-02-28"),
        ("某快餐品牌卫生问题", "食品安全", "抖音", 0.90, 2.9, 85.0, 2,
         "停业整顿涉事门店，全员卫生培训，安装明厨亮灶监控，邀请消费者参观后厨", "2024-06-01"),
        ("某汽车品牌刹车失灵", "产品质量", "新闻媒体", 0.95, 3.0, 92.0, 1,
         "全球召回，联合第三方检测机构调查，CEO公开致歉，更换供应商，加强质检", "2024-04-05"),
        ("某航空公司超售事件", "服务纠纷", "微博", 0.70, 2.5, 65.0, 3,
         "调整超售政策，提高补偿标准，员工沟通培训，设立旅客权益保障基金", "2024-07-12"),
        ("某化妆品过敏事件", "产品质量", "小红书", 0.65, 2.0, 55.0, 12,
         "下架问题批次，公布成分检测报告，完善过敏提示标签，赔偿医疗费用", "2024-03-28"),
        # === 2025-2026 年新增案例（覆盖更多行业） ===
        ("某辣条品牌口味争议", "口味争议", "微博", 0.45, 1.5, 28.0, 48,
         "收集消费者反馈优化配方，推出试吃装降低尝试门槛，联合美食KOL重新评测，在电商平台逐条回复差评并提供补偿", "2025-06-15"),
        ("某咖啡品牌价格上调事件", "价格争议", "小红书", 0.55, 1.8, 35.0, 24,
         "发布调价说明解释成本结构，推出会员锁定原价权益，加强'品质对得起价格'的内容营销，保留平价产品线满足不同消费层级", "2025-09-01"),
        ("某新茶饮品牌虚假宣传事件", "虚假宣传", "微博", 0.68, 2.2, 52.0, 6,
         "下架争议广告语，公布产品真实配料表和营养成分，CEO在社交媒体直接回应消费者质疑，邀请第三方检测每款产品成分并公示", "2025-11-20"),
        ("某短视频平台主播售假事件", "产品质量", "抖音", 0.82, 2.8, 75.0, 2,
         "永久封禁涉事主播账号，先行赔付消费者损失，升级商家保证金制度，引入AI+人工双重商品审核机制，每月发布平台治理报告", "2026-01-10"),
        ("某运动品牌代言人争议", "代言风险", "微博", 0.72, 2.5, 68.0, 4,
         "24小时内解除争议代言合同，发布致歉声明，建立代言人背景审查委员会，将代言人选择标准对外公开，邀请消费者参与品牌价值观共建", "2026-02-28"),
        ("某连锁健身房跑路风波", "服务纠纷", "抖音", 0.88, 2.6, 82.0, 3,
         "创始人出面承诺不会跑路并公布财务状况，引入第三方资金托管保障会员权益，按次付费替代年卡模式降低消费者风险，行业率先推出'7天无理由退卡'", "2026-03-15"),
        ("某矿泉水水源污染质疑", "食品安全", "新闻媒体", 0.78, 2.8, 70.0, 5,
         "立即公开水源地检测报告和生产线监控视频，邀请媒体和消费者代表实地考察水源地，第三方机构每月检测并公示，升级灌装车间洁净标准", "2026-04-01"),
        ("某宠物食品致宠物死亡事件", "食品安全", "小红书", 0.95, 2.9, 95.0, 1,
         "立即下架全批次产品并公告召回，全额退款+宠物医疗赔偿+精神损失补偿，送检第三方实验室公布原料来源和检测结果，引入宠物食品安全官制度全程监督品控", "2026-05-20"),
        ("某生鲜平台大数据杀熟", "价格争议", "知乎", 0.62, 2.0, 48.0, 12,
         "发布公开信承认算法存在缺陷，调整价格策略取消差异化定价，建立价格透明机制公示历史价格曲线，主动退还老用户高于新用户的差额部分", "2026-06-10"),
        ("某网约车平台数据泄露", "数据安全", "新闻媒体", 0.90, 3.0, 88.0, 3,
         "立即通知受影响用户并提供免费信用监控服务，聘请第三方安全公司全面审计，升级数据加密和访问控制体系，CEO公开致歉并承诺每年发布数据安全报告", "2026-07-01"),
        ("某家电品牌售后踢皮球", "服务投诉", "微博", 0.58, 1.8, 40.0, 18,
         "建立全国统一售后服务平台取消外包模式，承诺'只换不修'降低消费者等待成本，设立CEO直通投诉邮箱24小时响应，定期回访维修用户满意度", "2026-07-20"),
        ("某教培机构退费难", "服务纠纷", "知乎", 0.76, 2.3, 65.0, 8,
         "开通退费绿色通道承诺7个工作日内到账，引入第三方资金监管机构保障预付费安全，按课时付费替代预付费模式，公开发布财务状况和退费进度", "2026-08-05"),
    ]
    for row in history_data:
        cur.execute(
            "INSERT INTO history_records (event_name, category, platform, negative_score, "
            "media_weight, risk_index, hot_search_hours, resolution, occurred_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", row
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"[数据库] 样例数据填充完成：媒体权重12条 + 历史案例{len(history_data)}条")
