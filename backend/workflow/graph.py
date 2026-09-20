# ============================================================
# backend/workflow/graph.py - LangGraph工作流构建 + 执行入口
# ============================================================

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from backend.models import (
    SentimentItem, SentimentResult, CompetitorAnalysis,
    PRStrategy, HistoricalCase, RiskLevel, Platform, SentimentLabel,
)
from backend.workflow.nodes import (
    node_crawler, node_sentiment_analysis,
    node_rag_search, node_mysql_query,
    node_competitor_analysis, node_risk_calculation, node_pr_generation,
)


# ==================== State类型定义 ====================

class WorkflowState(TypedDict):
    """LangGraph全局状态（所有节点共享）"""
    query_text: str
    brand_name: str
    simulate_count: int
    raw_sentiments: List[SentimentItem]
    sentiment_results: List[SentimentResult]
    similar_cases: List[HistoricalCase]
    media_weight_total: float
    audience_reach: int
    competitor_results: List[CompetitorAnalysis]
    pr_strategies: List[PRStrategy]
    risk_index: float
    risk_level: RiskLevel
    trend_forecast: dict
    hot_search_probability: float
    negative_count: int
    positive_count: int
    negative_ratio: float


# ==================== 工作流构建 ====================

def build_workflow() -> StateGraph:
    """
    构建StateGraph工作流

    执行流程（=== 标注并行 ===）：
      [START] → [爬虫] → [情感分析]
                              ├──→ [RAG检索]   === 并行 ===
                              └──→ [MySQL查询]  === 并行 ===
                                    ↓
                              [竞品反应] → [风险计算] → [公关策略] → [END]
    """
    workflow = StateGraph(WorkflowState)

    # 注册7个节点
    workflow.add_node("crawler", node_crawler)
    workflow.add_node("sentiment", node_sentiment_analysis)
    workflow.add_node("rag_search", node_rag_search)
    workflow.add_node("mysql_query", node_mysql_query)
    workflow.add_node("competitor", node_competitor_analysis)
    workflow.add_node("risk_calc", node_risk_calculation)
    workflow.add_node("pr_generation", node_pr_generation)

    # 定义流转边
    workflow.set_entry_point("crawler")
    workflow.add_edge("crawler", "sentiment")

    # === 并行分支：RAG + MySQL 同时执行 ===
    workflow.add_edge("sentiment", "rag_search")
    workflow.add_edge("sentiment", "mysql_query")

    # 汇总到竞品分析
    workflow.add_edge("rag_search", "competitor")
    workflow.add_edge("mysql_query", "competitor")

    # 后续串行链路
    workflow.add_edge("competitor", "risk_calc")
    workflow.add_edge("risk_calc", "pr_generation")
    workflow.add_edge("pr_generation", END)

    return workflow


# ==================== 一键执行入口 ====================

def run_crisis_analysis(query_text: str, brand_name: str,
                        simulate_count: int = 20) -> dict:
    """
    一键运行完整舆情危机分析工作流

    参数:
        query_text: 搜索关键词
        brand_name: 品牌名称
        simulate_count: 模拟舆情条数

    返回:
        dict: 完整分析结果，可直接序列化返回前端
    """
    workflow = build_workflow()
    app = workflow.compile()

    # 初始状态
    initial_state: WorkflowState = {
        "query_text": query_text,
        "brand_name": brand_name,
        "simulate_count": simulate_count,
        "raw_sentiments": [],
        "sentiment_results": [],
        "similar_cases": [],
        "media_weight_total": 1.0,
        "audience_reach": 0,
        "competitor_results": [],
        "pr_strategies": [],
        "risk_index": 0.0,
        "risk_level": RiskLevel.LOW,
        "trend_forecast": {},
        "hot_search_probability": 0.0,
        "negative_count": 0,
        "positive_count": 0,
        "negative_ratio": 0.0,
    }

    # 执行工作流
    final_state = app.invoke(initial_state)

    # === 将分析结果存入向量库（下次RAG检索能搜到） ===
    _save_to_vector_db(brand_name, query_text, final_state)

    # === 格式化为JSON可序列化的dict ===
    result = _format_result(query_text, brand_name, final_state)

    print(f"\n{'='*60}")
    print(f"  舆情危机分析完成！")
    print(f"  品牌: {brand_name} | 风险等级: {final_state['risk_level'].value}")
    print(f"  风险指数: {final_state['risk_index']} | 热搜概率: {final_state['hot_search_probability']}%")
    print(f"{'='*60}\n")

    return result


def _save_to_vector_db(brand_name: str, query_text: str, state: dict):
    """提取分析结果的关键信息，存入向量库供后续RAG检索"""
    try:
        from backend.qdrant_client import save_analysis

        # 从情感分析结果中提取负面关键词
        sentiment_kw = []
        for r in state.get("sentiment_results", []):
            sentiment_kw.extend(r.negative_keywords if hasattr(r, 'negative_keywords') else [])

        # 从公关策略中提取行动步骤
        pr_actions = []
        for p in state.get("pr_strategies", []):
            pr_actions.extend(p.action_steps if hasattr(p, 'action_steps') else [])

        # 已有相似案例的标题
        case_titles = [c.title for c in state.get("similar_cases", [])]

        # 推断品类
        category = ""
        cases = state.get("similar_cases", [])
        if cases and hasattr(cases[0], 'category'):
            category = cases[0].category
        if not category:
            # 从舆情内容推断
            for s in state.get("raw_sentiments", []):
                if hasattr(s, 'content') and s.content:
                    for kw, cat in [("辣条", "辣条零食"), ("零食", "休闲零食"), ("手机", "手机数码"),
                                   ("奶茶", "茶饮"), ("咖啡", "咖啡饮品")]:
                        if kw in s.content:
                            category = cat
                            break
                if category:
                    break

        save_analysis(
            brand_name=brand_name,
            category=category,
            risk_level=state.get("risk_level").value if hasattr(state.get("risk_level"), 'value') else str(state.get("risk_level", "未知")),
            risk_index=state.get("risk_index", 0),
            negative_ratio=state.get("negative_ratio", 0),
            sentiment_keywords=sentiment_kw,
            pr_actions=pr_actions,
            similar_case_titles=case_titles,
        )
    except Exception as e:
        print(f"[向量库] 保存分析结果失败（不影响主流程）: {e}")


def _format_result(query_text: str, brand_name: str, state: dict) -> dict:
    """将WorkflowState转为前端可消费的纯dict格式"""
    return {
        "query": query_text,
        "brand": brand_name,
        "summary": {
            "risk_index": state["risk_index"],
            "risk_level": state["risk_level"].value,
            "hot_search_probability": state["hot_search_probability"],
            "negative_count": state["negative_count"],
            "positive_count": state["positive_count"],
            "total_sentiments": len(state["raw_sentiments"]),
            "negative_ratio": round(state.get("negative_ratio", 0) * 100, 1),
        },
        "trend_forecast": state["trend_forecast"],
        "sentiments": [
            {
                "id": s.id, "platform": s.platform.value,
                "content": s.content, "author": s.author,
                "publish_time": s.publish_time.isoformat(),
                "likes": s.likes, "shares": s.shares,
                "comments": s.comments, "media_weight": s.media_weight,
            }
            for s in state["raw_sentiments"]
        ],
        "sentiment_analysis": [
            {
                "original_id": r.original_id, "score": r.score,
                "label": r.label.value, "negative_keywords": r.negative_keywords,
                "risk_tags": r.risk_tags,
            }
            for r in state["sentiment_results"]
        ],
        "similar_cases": [
            {
                "case_id": c.case_id, "title": c.title,
                "category": c.category, "description": c.description,
                "resolution": c.resolution, "risk_level": c.risk_level.value,
                "score": c.score,
            }
            for c in state["similar_cases"]
        ],
        "competitor_analysis": [
            {
                "competitor_name": c.competitor_name,
                "action_type": c.action_type,
                "probability": c.probability,
                "description": c.description,
                "historical_case": c.historical_case,
            }
            for c in state["competitor_results"]
        ],
        "pr_strategies": [
            {
                "risk_level": p.risk_level.value, "urgency": p.urgency,
                "talking_points": p.talking_points,
                "action_steps": p.action_steps,
                "suggested_statement": p.suggested_statement,
            }
            for p in state["pr_strategies"]
        ],
        "media_info": {
            "weight_total": state["media_weight_total"],
            "audience_reach": state["audience_reach"],
        },
    }
