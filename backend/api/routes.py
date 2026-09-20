# ============================================================
# backend/api/routes.py - FastAPI路由定义
# 所有API端点集中管理，main.py中注册
# ============================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

from backend.models import AnalyzeRequest, AnalyzeResponse
from backend.workflow import run_crisis_analysis
from backend.database import get_all_media_weights, get_all_history_records
from backend.llm_client import check_connection as check_llm, is_llm_available
from backend.search import search_news

# 创建路由分组
router = APIRouter()


# ==================== 系统首页 ====================

@router.get("/", tags=["系统"])
async def root():
    """系统首页"""
    return {
        "name": "舆情危机预警与智能公关推演系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "frontend": "/app",
    }


# ==================== 核心分析接口 ====================

@router.post("/api/analyze", response_model=AnalyzeResponse, tags=["核心分析"])
async def analyze_crisis(request: AnalyzeRequest):
    """
    ## 一键舆情危机分析

    触发完整LangGraph工作流（7个节点，2个并行分支）：
    1. 爬虫Agent → 模拟抓取舆情
    2. 情感分析Agent → NLP情感打分
    3. RAG检索 + MySQL查询 → **并行执行**
    4. 竞品反应Agent → 推演竞品动作
    5. 风险计算 → 综合风险指数
    6. 公关策略Agent → 输出分级应对方案
    """
    try:
        result = run_crisis_analysis(
            query_text=request.query,
            brand_name=request.brand_name,
            simulate_count=request.simulate_count,
        )
        return AnalyzeResponse(
            success=True,
            data=result,
            message=f"分析完成：风险等级 {result['summary']['risk_level']}，风险指数 {result['summary']['risk_index']}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


# ==================== 数据查询接口 ====================

@router.get("/api/history", tags=["数据查询"])
async def get_history():
    """获取MySQL中所有历史舆情记录"""
    try:
        records = get_all_history_records()
        for r in records:
            if hasattr(r.get("occurred_at"), "isoformat"):
                r["occurred_at"] = r["occurred_at"].isoformat()
        return {"success": True, "data": records, "count": len(records)}
    except Exception as e:
        return {"success": False, "data": [], "message": str(e)}


@router.get("/api/media-weights", tags=["数据查询"])
async def get_media_weights():
    """获取所有媒体权重数据"""
    try:
        records = get_all_media_weights()
        return {"success": True, "data": records, "count": len(records)}
    except Exception as e:
        return {"success": False, "data": [], "message": str(e)}


@router.get("/api/stats", tags=["数据查询"])
async def get_stats():
    """获取系统统计数据（风险分布、事件总量等）"""
    try:
        history = get_all_history_records()
        media = get_all_media_weights()
        return {
            "success": True,
            "data": {
                "total_history_events": len(history),
                "total_media_sources": len(media),
                "risk_distribution": {
                    "critical": sum(1 for h in history if h.get("risk_index", 0) >= 75),
                    "high": sum(1 for h in history if 50 <= h.get("risk_index", 0) < 75),
                    "medium": sum(1 for h in history if 25 <= h.get("risk_index", 0) < 50),
                    "low": sum(1 for h in history if h.get("risk_index", 0) < 25),
                },
            }
        }
    except Exception as e:
        return {"success": False, "data": {}, "message": str(e)}


# ==================== 报告导出 ====================

class ExportRequest(BaseModel):
    brand_name: str = Field(default="XX品牌")
    query: str = Field(default="")
    include_cases: bool = Field(default=True)
    include_strategies: bool = Field(default=True)


@router.post("/api/export-report", tags=["报告导出"])
async def export_report(request: ExportRequest):
    """
    ## 导出公关处理报告

    整合舆情数据 + 相似案例 + 应对策略 → 生成完整报告文本
    """
    try:
        analysis = run_crisis_analysis(
            query_text=request.query or f"{request.brand_name}舆情",
            brand_name=request.brand_name,
            simulate_count=20,
        )

        summary = analysis["summary"]
        report_lines = [
            "=" * 60,
            "  舆情危机预警与公关处理报告",
            "=" * 60, "",
            f"品牌名称：{request.brand_name}",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"搜索关键词：{request.query or '未指定'}", "",
            "-" * 40, "一、风险总览", "-" * 40,
            f"  综合风险指数：{summary['risk_index']}/100",
            f"  风险等级：{summary['risk_level']}",
            f"  热搜概率：{summary['hot_search_probability']}%",
            f"  负面舆情占比：{summary['negative_ratio']}%",
            f"  分析舆情总数：{summary['total_sentiments']} 条", "",
            "-" * 40, "二、热度走势预测", "-" * 40,
        ]

        trend = analysis.get("trend_forecast", {})
        for label, key in [("2小时后", "2h"), ("12小时后", "12h"), ("24小时后", "24h")]:
            report_lines.append(f"  {label}预估风险值：{trend.get(key, 'N/A')}")

        if request.include_cases:
            report_lines.extend(["", "-" * 40, "三、相似历史案例", "-" * 40])
            for i, case in enumerate(analysis.get("similar_cases", [])[:3], 1):
                report_lines.append(f"  {i}. {case['title']}（相似度：{case['score']}）")
                report_lines.append(f"     分类：{case['category']} | 风险等级：{case['risk_level']}")
                report_lines.append(f"     处理方案：{case['resolution'][:100]}...")

        if request.include_strategies:
            report_lines.extend(["", "-" * 40, "四、公关应对策略", "-" * 40])
            for i, strategy in enumerate(analysis.get("pr_strategies", []), 1):
                report_lines.append(f"  {i}. [{strategy['urgency']}]")
                report_lines.append(f"     话术要点：{'；'.join(strategy['talking_points'][:3])}")
                if strategy.get("suggested_statement"):
                    report_lines.append(f"     建议声明：{strategy['suggested_statement'][:120]}...")

        report_lines.extend(["", "=" * 60, "报告结束", "=" * 60])
        report_text = "\n".join(report_lines)

        return {"success": True, "report": report_text, "data": analysis, "message": "报告生成成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")


# ==================== 健康检查 ====================

@router.get("/api/health", tags=["系统"])
async def health_check():
    """系统健康检查（含LLM状态）"""
    llm_status = check_llm()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "fastapi": "running",
            "langgraph": "ready",
            "llm": llm_status,
            "qdrant": "check /api/analyze to verify",
            "mysql": "check /api/history to verify",
            "search": "configured" if search_news else "pending",
        }
    }
