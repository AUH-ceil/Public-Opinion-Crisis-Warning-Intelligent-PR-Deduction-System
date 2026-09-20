# ============================================================
# backend/risk_calculator.py - 舆情危机风险指数计算器
# 核心公式：风险指数 = 媒体权重 × 负面情感分绝对值 × 传播热度
# 所有计算集中在单一函数，逻辑直接透明
# ============================================================

import math
from typing import List
from backend.models import SentimentResult, RiskLevel


def calculate_risk_index(
    sentiment_results: List[SentimentResult],
    media_weight_total: float,
    audience_reach: int,
) -> float:
    """
    综合风险指数计算（0-100分制）

    公式拆解：
    1. 负面情感强度 = 所有负面舆情的情感分值绝对值之和 / 总数
       （只有负面舆情参与计算，正向和中性不计入）
    2. 传播热度 = log10(触达人数 + 1) / 10
       （用对数平滑，避免大V数据导致指数爆炸）
    3. 风险指数 = min(100, 负面情感强度 × 媒体权重 × 传播热度 × 100)

    参数：
        sentiment_results: 情感分析结果列表
        media_weight_total: MySQL查询的加权媒体总权重
        audience_reach: 预估触达总人数

    返回：
        0-100之间的风险指数浮点数
    """
    if not sentiment_results:
        return 0.0

    # ===== 第一步：计算负面情感平均强度 =====
    negative_scores = [
        abs(r.score) for r in sentiment_results
        if r.score < 0  # 只取负面（score < 0）
    ]

    if not negative_scores:
        return 0.0  # 没有负面舆情，风险为0

    avg_negative_intensity = sum(negative_scores) / len(sentiment_results)

    # ===== 第二步：计算传播热度（对数平滑） =====
    # 基于触达人数，取对数避免数值爆炸
    # 10万人触达 ≈ 0.5 / 100万人触达 ≈ 0.6 / 1亿人触达 ≈ 0.8
    spread_heat = math.log10(max(audience_reach, 1) + 1) / 10.0
    spread_heat = min(spread_heat, 1.0)  # 上限1.0

    # ===== 第三步：综合计算 =====
    # 媒体权重一般 1.0~3.0（头部媒体权重高）
    # 负面情感强度 0~1.0
    # 传播热度 0~1.0
    raw_risk = avg_negative_intensity * media_weight_total * spread_heat * 100

    # 限制在 0-100 范围
    risk_index = min(100.0, max(0.0, round(raw_risk, 1)))

    return risk_index


def get_risk_level(risk_index: float) -> RiskLevel:
    """
    根据风险指数判定预警等级

    等级划分：
        0-25    → 低风险（日常监控即可）
        25-50   → 中风险（密切关注，准备预案）
        50-75   → 高风险（立即启动公关响应）
        75-100  → 特级风险（全员应急，最高优先级）
    """
    if risk_index >= 75:
        return RiskLevel.CRITICAL
    elif risk_index >= 50:
        return RiskLevel.HIGH
    elif risk_index >= 25:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def forecast_trend(current_risk: float, negative_count: int,
                   total_count: int) -> dict:
    """
    热度走势预测（简易公式推演）
    模拟2小时/12小时/24小时后的舆情热度变化

    逻辑：
    - 负面占比越高，持续发酵概率越大
    - 按时间递增，假设自然衰减系数递减（越久越可能被其他热点替代）
    - 但如果有大量负面，衰减慢甚至逆势增长
    """
    if total_count == 0:
        return {"2h": 0, "12h": 0, "24h": 0}

    negative_ratio = negative_count / total_count

    # 衰减/增长系数（负面占比越高，增长越猛）
    # 负面占比 > 60%：持续发酵，不降反升
    if negative_ratio > 0.6:
        multipliers = {"2h": 1.05, "12h": 1.15, "24h": 1.25}
    elif negative_ratio > 0.3:
        multipliers = {"2h": 1.0, "12h": 1.05, "24h": 0.95}
    else:
        multipliers = {"2h": 0.95, "12h": 0.85, "24h": 0.7}

    return {
        "2h": round(min(100, current_risk * multipliers["2h"]), 1),
        "12h": round(min(100, current_risk * multipliers["12h"]), 1),
        "24h": round(min(100, current_risk * multipliers["24h"]), 1),
    }


def forecast_hot_search_probability(risk_index: float, negative_ratio: float) -> float:
    """
    预判登上热搜的概率（0-100%）

    简易模型：
    - 风险指数权重 60%
    - 负面占比权重 40%
    - 指数超过60分时，热搜概率显著上升
    """
    if risk_index >= 80:
        base_prob = 85 + (risk_index - 80)  # 80分以上，热搜概率85%+
    elif risk_index >= 60:
        base_prob = 50 + (risk_index - 60) * 1.75  # 60-80分，线性增长到85%
    elif risk_index >= 40:
        base_prob = 15 + (risk_index - 40) * 1.75  # 40-60分，线性增长到50%
    else:
        base_prob = risk_index * 0.375  # 40分以下，缓慢增长

    # 负面占比加成
    negative_bonus = negative_ratio * 15  # 负面占比100%加15%

    return round(min(100, max(0, base_prob + negative_bonus)), 1)
