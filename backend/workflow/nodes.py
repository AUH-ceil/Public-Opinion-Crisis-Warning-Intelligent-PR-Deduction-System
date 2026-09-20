# ============================================================
# backend/workflow/nodes.py - LangGraph工作流节点函数
# 每个节点 = 一个LangGraph执行单元
# 节点间通过WorkflowState共享数据
# ============================================================

from typing import List
from backend.models import (
    SentimentItem, SentimentResult, CompetitorAnalysis,
    PRStrategy, HistoricalCase, RiskLevel,
)
from backend.agents.crawler_agent import crawler_agent
from backend.agents.sentiment_agent import sentiment_agent
from backend.agents.competitor_agent import competitor_agent
from backend.agents.pr_strategy_agent import pr_strategy_agent
from backend.qdrant_client import search_similar_cases
from backend.database import get_all_media_weights, get_total_audience_reach
from backend.risk_calculator import (
    calculate_risk_index, get_risk_level,
    forecast_trend, forecast_hot_search_probability,
)


# ==================== 节点1：爬虫Agent ====================

def node_crawler(state: dict) -> dict:
    """模拟从各平台抓取舆情数据"""
    items = crawler_agent(
        query_text=state["query_text"],
        brand_name=state["brand_name"],
        count=state["simulate_count"],
    )
    return {"raw_sentiments": items}


# ==================== 节点2：情感分析Agent ====================

def node_sentiment_analysis(state: dict) -> dict:
    """对原始舆情逐条进行情感打分和风险标签提取"""
    if not state.get("raw_sentiments"):
        return {}

    results = sentiment_agent(state["raw_sentiments"])

    # 统计正/负/中性分布
    neg_count = sum(1 for r in results if r.label.value == "负面")
    pos_count = sum(1 for r in results if r.label.value == "正向")
    total = len(results) if results else 1

    return {
        "sentiment_results": results,
        "negative_count": neg_count,
        "positive_count": pos_count,
        "negative_ratio": neg_count / total,
    }


# ==================== 节点3：RAG历史案例检索 ====================

def node_rag_search(state: dict) -> dict:
    """
    Qdrant向量库语义检索 + MySQL结构化历史数据合并。
    两路数据互补：Qdrant负责语义相似度，MySQL补充精确分类匹配。
    === 此节点与 node_mysql_query 并行执行 ===
    """
    query = f"{state['brand_name']} {state['query_text']}"
    all_cases = {}  # case_id → HistoricalCase，用于去重

    # ===== 来源1：Qdrant 语义检索 =====
    try:
        qdrant_results = search_similar_cases(query, limit=5)
        for c in qdrant_results:
            cid = str(c.get("case_id", ""))
            all_cases[cid] = HistoricalCase(
                case_id=cid,
                title=str(c.get("title", "")),
                category=str(c.get("category", "")),
                description=str(c.get("description", "")),
                resolution=str(c.get("resolution", "")),
                risk_level=RiskLevel(str(c.get("risk_level", "低"))),
                score=float(c.get("score", 0.0)),
            )
    except Exception as e:
        print(f"[RAG检索] Qdrant检索失败: {e}")

    # ===== 来源2：MySQL 结构化历史数据 =====
    try:
        from backend.database import get_history_by_category

        # 用已分析出的风险标签作为分类关键词
        categories_to_search = set()
        for r in state.get("sentiment_results", []):
            for tag in (r.risk_tags if hasattr(r, 'risk_tags') else []):
                # 映射风险标签到 MySQL 分类
                cat_map = {
                    "食品安全": "食品安全", "产品质量": "产品质量",
                    "服务投诉": "服务纠纷", "虚假宣传": "产品质量",
                    "价格争议": "产品质量",
                }
                mapped = cat_map.get(tag)
                if mapped:
                    categories_to_search.add(mapped)

        if not categories_to_search:
            categories_to_search.add("产品质量")  # 默认分类

        for cat in list(categories_to_search)[:3]:
            rows = get_history_by_category(cat)
            for row in rows[:3]:  # 每类最多取3条
                cid = f"mysql_{row.get('id', '')}"
                if cid not in all_cases:
                    title = str(row.get("event_name", ""))
                    resolution = str(row.get("resolution", ""))
                    all_cases[cid] = HistoricalCase(
                        case_id=cid,
                        title=title,
                        category=str(row.get("category", cat)),
                        description=f"{title}（{row.get('platform', '')}首发，风险指数{row.get('risk_index', 0)}）",
                        resolution=resolution,
                        risk_level=_mysql_risk_to_enum(row.get("risk_index", 0)),
                        score=0.6,  # MySQL 精确匹配给中等相似分
                    )

        print(f"[RAG检索] MySQL补充 {len([k for k in all_cases if k.startswith('mysql_')])} 条分类匹配案例")
    except Exception as e:
        print(f"[RAG检索] MySQL查询失败（不影响主流程）: {e}")

    # 按相似度分数降序排列
    similar_cases = sorted(all_cases.values(), key=lambda x: x.score, reverse=True)
    print(f"[RAG检索] 合并后共 {len(similar_cases)} 条历史案例（Qdrant+MySQL）")
    return {"similar_cases": similar_cases}


def _mysql_risk_to_enum(risk_index: float) -> RiskLevel:
    """MySQL 的 risk_index 数值 → RiskLevel 枚举"""
    if risk_index >= 75:
        return RiskLevel.CRITICAL
    elif risk_index >= 50:
        return RiskLevel.HIGH
    elif risk_index >= 25:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# ==================== 节点4：MySQL数据查询 ====================

def node_mysql_query(state: dict) -> dict:
    """
    获取媒体权重数据和受众触达预估
    === 此节点与 node_rag_search 并行执行 ===
    """
    try:
        media_data = get_all_media_weights()
        if media_data:
            avg_weight = sum(m["weight"] for m in media_data) / len(media_data)
        else:
            avg_weight = 1.0
        audience = get_total_audience_reach()
        print(f"[MySQL查询] 媒体数据 {len(media_data)} 条，权重 {avg_weight:.2f}，触达 {audience:,}")
    except Exception as e:
        avg_weight = 1.5
        audience = 1000000
        print(f"[MySQL查询] 连接失败，使用默认值: 权重={avg_weight}, 触达={audience:,}")

    return {
        "media_weight_total": round(avg_weight, 2),
        "audience_reach": audience,
    }


# ==================== 节点5：竞品反应Agent ====================

def node_competitor_analysis(state: dict) -> dict:
    """基于舆情态势推演竞品可能的行动"""
    if not state.get("sentiment_results"):
        return {"competitor_results": []}

    temp_risk = state.get("risk_level", RiskLevel.MEDIUM)

    results = competitor_agent(
        brand_name=state["brand_name"],
        sentiment_results=state["sentiment_results"],
        risk_level=temp_risk,
        negative_ratio=state.get("negative_ratio", 0.4),
    )
    return {"competitor_results": results}


# ==================== 节点6：风险综合计算 ====================

def node_risk_calculation(state: dict) -> dict:
    """
    融合情感分析 + MySQL数据 → 计算风险指数 + 走势预测
    核心公式：媒体权重 × 负面情感分 × 传播热度
    """
    if not state.get("sentiment_results"):
        return {}

    # 计算风险指数
    risk_index = calculate_risk_index(
        sentiment_results=state["sentiment_results"],
        media_weight_total=state.get("media_weight_total", 1.0),
        audience_reach=state.get("audience_reach", 100000),
    )
    risk_level = get_risk_level(risk_index)

    # 热度走势预测
    trend = forecast_trend(
        current_risk=risk_index,
        negative_count=state.get("negative_count", 0),
        total_count=len(state["sentiment_results"]),
    )

    # 热搜概率
    hot_prob = forecast_hot_search_probability(
        risk_index=risk_index,
        negative_ratio=state.get("negative_ratio", 0.4),
    )

    print(f"[风险计算] 指数={risk_index} 等级={risk_level.value} 热搜概率={hot_prob}%")
    return {
        "risk_index": risk_index,
        "risk_level": risk_level,
        "trend_forecast": trend,
        "hot_search_probability": hot_prob,
    }


# ==================== 节点7：公关策略生成 ====================

def node_pr_generation(state: dict) -> dict:
    """
    综合所有前置结果，生成分级公关应对方案。

    传递给 PR Agent 的完整上下文：
      - RAG 相似案例（核心参考）
      - 竞品动作推演
      - 情感分析详情（消费者具体在抱怨什么）
      - 品牌品类信息（来自爬虫阶段或案例推断）
      - 风险等级 + 负面占比
    """
    # 尝试从 raw_sentiments 推断品牌品类
    brand_category = _infer_brand_category(state)

    pr_strategies = pr_strategy_agent(
        brand_name=state["brand_name"],
        risk_level=state.get("risk_level", RiskLevel.MEDIUM),
        similar_cases=state.get("similar_cases", []),
        competitor_actions=state.get("competitor_results", []),
        negative_ratio=state.get("negative_ratio", 0.4),
        sentiment_results=state.get("sentiment_results", []),
        brand_category=brand_category,
    )
    return {"pr_strategies": pr_strategies}


def _infer_brand_category(state: dict) -> str:
    """从RAG案例或舆情内容推断品牌品类"""
    # 优先从RAG案例获取
    cases = state.get("similar_cases", [])
    if cases and cases[0].category:
        return cases[0].category

    # 从舆情文本推断（简单关键词匹配作为降级）
    texts = [s.content for s in state.get("raw_sentiments", []) if hasattr(s, 'content')]
    combined = " ".join(texts[:5])

    cat_keywords = {
        "辣条": "辣条零食", "零食": "休闲零食", "奶茶": "茶饮", "咖啡": "咖啡饮品",
        "手机": "手机数码", "电脑": "电子产品", "汽车": "汽车", "酒店": "酒店服务",
        "外卖": "外卖配送", "快递": "快递物流", "化妆品": "化妆品", "药品": "医药",
        "饮料": "饮品", "食品": "食品", "餐饮": "餐饮服务",
    }
    for kw, cat in cat_keywords.items():
        if kw in combined:
            return cat

    return "消费品"
