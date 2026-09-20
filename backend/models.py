# ============================================================
# backend/models.py - 数据模型定义（Pydantic + 数据库映射）
# 统一管理所有数据结构，方便各模块引用
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ==================== 枚举定义 ====================

class RiskLevel(str, Enum):
    """危机预警等级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "特级"


class SentimentLabel(str, Enum):
    """情感标签"""
    POSITIVE = "正向"
    NEUTRAL = "中性"
    NEGATIVE = "负面"


class Platform(str, Enum):
    """模拟发布平台"""
    WEIBO = "微博"
    DOUYIN = "抖音"
    REDBOOK = "小红书"
    NEWS = "新闻媒体"
    TIEBA = "贴吧"
    ZHIHU = "知乎"


# ==================== 舆情数据模型 ====================

class SentimentItem(BaseModel):
    """单条舆情数据"""
    id: str = Field(default="", description="舆情唯一ID")
    platform: Platform = Field(description="发布平台")
    content: str = Field(description="舆情文本内容")
    author: str = Field(default="匿名用户", description="发布者")
    publish_time: datetime = Field(default_factory=datetime.now, description="发布时间")
    likes: int = Field(default=0, description="点赞数")
    shares: int = Field(default=0, description="转发数")
    comments: int = Field(default=0, description="评论数")
    media_weight: float = Field(default=1.0, description="媒体权重系数")


class SentimentResult(BaseModel):
    """情感分析结果"""
    original_id: str = Field(description="对应舆情ID")
    score: float = Field(default=0.0, ge=-1.0, le=1.0, description="情感分值 -1负面~+1正向")
    label: SentimentLabel = Field(description="情感标签")
    negative_keywords: List[str] = Field(default_factory=list, description="负面关键词")
    risk_tags: List[str] = Field(default_factory=list, description="风险标签")


class CompetitorAnalysis(BaseModel):
    """竞品反应推演结果"""
    competitor_name: str = Field(description="竞品名称")
    action_type: str = Field(description="动作类型：抹黑/借势/沉默/蹭热度")
    probability: float = Field(default=0.0, ge=0.0, le=1.0, description="发生概率")
    description: str = Field(description="详细推演描述")
    historical_case: str = Field(default="", description="历史类似案例")


class PRStrategy(BaseModel):
    """公关应对策略"""
    risk_level: RiskLevel = Field(description="风险等级")
    urgency: str = Field(description="紧急程度：紧急/常规/长期")
    talking_points: List[str] = Field(default_factory=list, description="话术要点")
    action_steps: List[str] = Field(default_factory=list, description="处理步骤")
    suggested_statement: str = Field(default="", description="建议声明文案")


class HistoricalCase(BaseModel):
    """历史舆情案例（Qdrant向量库存储+检索）"""
    case_id: str = Field(description="案例ID")
    title: str = Field(description="案例标题")
    category: str = Field(description="分类：食品安全/服务纠纷/产品质量等")
    description: str = Field(description="事件描述")
    resolution: str = Field(description="处理方案")
    risk_level: RiskLevel = Field(description="风险等级")
    score: float = Field(default=0.0, description="相似度分数")


# ==================== LangGraph全局状态 ====================

class CrisisState(BaseModel):
    """
    LangGraph StateGraph 全局舆情状态
    === 所有Agent并行读写此状态，统一数据管理 ===
    """
    # 输入
    query_text: str = Field(default="", description="舆情查询文本")
    brand_name: str = Field(default="XX品牌", description="品牌名称")
    raw_sentiments: List[SentimentItem] = Field(default_factory=list, description="原始舆情列表")

    # 情感分析
    sentiment_results: List[SentimentResult] = Field(default_factory=list, description="情感分析结果")
    avg_sentiment_score: float = Field(default=0.0, description="平均情感分值")
    negative_count: int = Field(default=0, description="负面数量")
    positive_count: int = Field(default=0, description="正向数量")

    # RAG历史案例
    similar_cases: List[HistoricalCase] = Field(default_factory=list, description="相似案例")

    # MySQL媒体权重
    media_weight_total: float = Field(default=1.0, description="加权媒体总权重")
    audience_reach: int = Field(default=0, description="预估触达人数")

    # 竞品推演
    competitor_results: List[CompetitorAnalysis] = Field(default_factory=list, description="竞品反应")

    # 公关策略
    pr_strategies: List[PRStrategy] = Field(default_factory=list, description="公关策略")

    # 综合风险
    risk_index: float = Field(default=0.0, description="综合风险指数 0-100")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="风险等级")

    # 推演走势
    trend_forecast: dict = Field(default_factory=dict, description="热度走势 {2h, 12h, 24h}")
    hot_search_probability: float = Field(default=0.0, description="上热搜概率")


# ==================== API请求/响应 ====================

class AnalyzeRequest(BaseModel):
    """舆情分析请求"""
    query: str = Field(default="", description="搜索提示词（可选，留空则LLM自动分析品牌）")
    brand_name: str = Field(default="XX品牌", description="品牌名称（必填）")
    simulate_count: int = Field(default=20, ge=5, le=100, description="抓取条数")


class AnalyzeResponse(BaseModel):
    """分析响应"""
    success: bool = True
    data: Optional[dict] = None
    message: str = "分析完成"
