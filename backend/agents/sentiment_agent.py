# ============================================================
# Agent 2: 情感分析Agent — LLM 驱动版
# ============================================================
# 改造要点：
#   v1.0: 43个关键词词典 + 随机抖动 → 看不懂反讽，漏判严重
#   v2.0: LLM 逐条分析 → 理解语义、识别阴阳怪气、上下文感知
#
# 降级策略：
#   LLM 可用 → 批量 LLM 分析（精确）
#   LLM 不可用 → 回退到关键词词典（兜底，保留原逻辑）
# ============================================================

import json
import random
from typing import List, Tuple

from backend.models import SentimentItem, SentimentResult, SentimentLabel
from backend.llm_client import call_llm_json, is_llm_available


# ==================== LLM System Prompt ====================

SENTIMENT_SYSTEM_PROMPT = """你是资深舆情分析师，专精中文互联网内容情感分析。请分析给定文本，输出 JSON。

## 分析维度：
1. **score**（-1.0 ~ 1.0）：情感分值
   - -0.8~-1.0 极负面（安全事故、健康危害、欺诈、违法违规、强烈谴责）
   - -0.5~-0.8 明显负面（踩雷、避雷、难吃/难用、投诉、维权、曝光）
   - -0.2~-0.5 轻微负面（吐槽、小失望、不值这个价、不如预期）
   - -0.1~0.2 中性（客观评测、有好有坏、提问、事实陈述）
   - 0.2~0.6 正向（好吃、好用、推荐、种草、回购、满意）
   - 0.6~1.0 极正向（强烈安利、远超预期、忠实粉丝、自发辩护）
2. **label**："负面" / "中性" / "正向"
3. **negative_keywords**：从原文提取的负面关键词（正向时为空数组）
4. **risk_tags**：仅在内容涉及实质风险时填写，普通差评不填。从以下选择：
   ["食品安全","产品质量","服务投诉","虚假宣传","数据安全","价格争议","合规风险","劳动纠纷","环境污染","歧视/偏见","法律诉讼","高管丑闻"]
   注意：普通消费者说"难吃""不值""失望"不属于风险标签范畴，不要强行贴标签

## 特别注意：
- **消费评价场景**：理解"好吃""难吃""踩雷""回购""种草""拔草""真香""翻车"等消费评价术语
- **反讽识别**：表情包(🙃🐶💩)、"真是太棒了呢"+狗头 → 实际是负面
- **水军识别**：模板化好评、大量感叹号、无实质内容 → 中性偏低分
- **段子/玩梗**：看似调侃实则传播负面 → 判为负面，分值适度(-0.3~-0.5)
- **不要把普通差评当成危机**：单纯说"不好吃""不值"的风险标签为空数组

返回格式：{"score": 0.0, "label": "中性", "negative_keywords": [], "risk_tags": []}"""


# ==================== 原关键词词典（降级时用） ====================

NEGATIVE_DICT = {
    "翻车": -0.8, "避雷": -0.7, "投诉": -0.6, "退货": -0.5,
    "垃圾": -0.9, "恶心": -0.9, "差评": -0.7, "坑": -0.6,
    "假": -0.7, "骗": -0.8, "死": -0.9, "烂": -0.8,
    "坑爹": -0.8, "坑钱": -0.7, "垃圾产品": -0.9, "维权": -0.5,
    "爆雷": -0.8, "砸了": -0.7, "曝光": -0.6, "举报": -0.5,
    "失望": -0.5, "恶心到": -0.9, "受不了": -0.6, "控评": -0.7,
    "删评": -0.7, "水军": -0.5, "黑心": -0.8, "有毒": -0.9,
    "致癌": -0.95, "过敏": -0.7, "超标": -0.8, "劣质": -0.7,
    "污染": -0.8, "虚假": -0.7, "骗人": -0.8, "骗子": -0.8,
    "差": -0.4, "不满": -0.5, "气愤": -0.7, "可怕": -0.7,
    "严重": -0.5, "紧急": -0.5, "警惕": -0.6, "注意": -0.3,
    "问题": -0.3, "处理": -0.2,
}
POSITIVE_DICT = {
    "支持": 0.6, "相信": 0.5, "不错": 0.4, "喜欢": 0.5,
    "好": 0.4, "棒": 0.6, "推荐": 0.6, "赞": 0.5,
    "好评": 0.7, "满意": 0.6, "靠谱": 0.5, "优秀": 0.6,
    "良心": 0.7, "感谢": 0.5, "值得": 0.5, "放心": 0.5,
    "好样的": 0.7, "给力": 0.6, "yyds": 0.7, "爱了": 0.6,
}
RISK_TAG_MAP = {
    "食品安全": ["有毒", "致癌", "污染", "超标", "食品安全", "吃出", "变质",
              "过期", "虫子", "异物", "拉肚子", "食物中毒", "卫生"],
    "产品质量": ["劣质", "烂", "坏了", "电池", "爆炸", "自燃", "故障",
              "缺陷", "召回", "损坏", "裂", "碎", "断"],
    "服务投诉": ["客服", "态度", "投诉", "差评", "退货", "售后",
              "推诿", "踢皮球", "敷衍", "不理", "无人"],
    "虚假宣传": ["虚假", "骗", "夸大", "忽悠", "欺骗", "诱导",
              "假一赔", "假货", "冒牌", "山寨"],
    "数据安全": ["泄露", "隐私", "数据", "盗用", "窃取", "黑客"],
    "价格争议": ["坑钱", "贵", "涨价", "宰客", "天价", "暴利"],
}


# ==================== 情感分析主函数 ====================

def sentiment_agent(items: List[SentimentItem]) -> List[SentimentResult]:
    """
    【情感分析Agent v2.0】
    LLM 逐条分析舆情情感，覆盖反讽/段子/水军等复杂场景。

    参数：
        items: 原始舆情列表

    返回：
        List[SentimentResult]: 情感分析结果（分值、标签、关键词、风险标签）
    """
    if not items:
        return []

    # 检测 LLM 可用性
    if is_llm_available():
        results = _sentiment_with_llm(items)
    else:
        print("[情感分析Agent] LLM不可用，降级为关键词词典分析")
        results = _sentiment_rule_based(items)

    # 统计汇总
    neg_count = sum(1 for r in results if r.label == SentimentLabel.NEGATIVE)
    pos_count = sum(1 for r in results if r.label == SentimentLabel.POSITIVE)
    neu_count = sum(1 for r in results if r.label == SentimentLabel.NEUTRAL)
    avg_score = round(sum(r.score for r in results) / len(results), 2) if results else 0

    print(f"[情感分析Agent] 分析完成：负面{neg_count} 正向{pos_count} 中性{neu_count} 均分{avg_score}")

    return results


# ==================== LLM 模式（精确分析） ====================

def _sentiment_with_llm(items: List[SentimentItem]) -> List[SentimentResult]:
    """用 LLM 逐条分析情感（支持 JSON Mode 强制输出）"""
    results = []

    for i, item in enumerate(items):
        # 构建丰富的上下文
        user_prompt = _build_prompt(item)

        try:
            parsed = call_llm_json(
                system_prompt=SENTIMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
            )

            if "error" in parsed:
                # LLM JSON 解析失败，用规则引擎兜底
                result = _rule_single(item)
                results.append(result)
                continue

            score = float(parsed.get("score", 0.0))
            score = max(-1.0, min(1.0, score))  # 确保在范围内

            label_str = parsed.get("label", "中性")
            if label_str not in ("正向", "负面", "中性"):
                label_str = "中性"
            label = SentimentLabel(label_str)

            negative_keywords = list(parsed.get("negative_keywords", []))[:10]
            risk_tags = list(parsed.get("risk_tags", []))[:5]

            result = SentimentResult(
                original_id=item.id,
                score=round(score, 2),
                label=label,
                negative_keywords=negative_keywords,
                risk_tags=risk_tags,
            )
            results.append(result)

        except Exception as e:
            print(f"[情感分析Agent] 第{i}条LLM分析失败: {e}，使用规则引擎兜底")
            result = _rule_single(item)
            results.append(result)

    return results


def _build_prompt(item: SentimentItem) -> str:
    """构建单条情感分析的 user prompt"""
    platform = item.platform.value if hasattr(item.platform, 'value') else str(item.platform)
    return f"""平台：{platform}
作者：{item.author}
互动：{item.likes}赞 {item.shares}转 {item.comments}评
内容：
{item.content}"""


# ==================== 规则引擎模式（降级/兜底） ====================

def _sentiment_rule_based(items: List[SentimentItem]) -> List[SentimentResult]:
    """原版关键词词典分析（LLM不可用时的降级方案）"""
    return [_rule_single(item) for item in items]


def _rule_single(item: SentimentItem) -> SentimentResult:
    """单条规则引擎分析（保留原版逻辑）"""
    content = item.content or ""

    total_score = 0.0
    hit_count = 0
    negative_words = []
    risk_tags = set()

    # 负面词典扫描
    for word, score in NEGATIVE_DICT.items():
        if word in content:
            total_score += score
            hit_count += 1
            negative_words.append(word)
            for tag, keywords in RISK_TAG_MAP.items():
                if word in keywords:
                    risk_tags.add(tag)

    # 正向词典扫描
    for word, score in POSITIVE_DICT.items():
        if word in content:
            total_score += score * 0.7
            hit_count += 1

    # 计算最终分值
    if hit_count > 0:
        final_score = max(-1.0, min(1.0, total_score / max(hit_count, 1)))
    else:
        final_score = -0.1 + random.uniform(-0.1, 0.2)

    final_score += random.uniform(-0.1, 0.1)
    final_score = max(-1.0, min(1.0, final_score))

    label = _get_sentiment_label(final_score)

    return SentimentResult(
        original_id=item.id,
        score=round(final_score, 2),
        label=label,
        negative_keywords=negative_words[:10],
        risk_tags=list(risk_tags)[:3],
    )


def _get_sentiment_label(score: float) -> SentimentLabel:
    """根据分值判定情感标签"""
    if score > 0.15:
        return SentimentLabel.POSITIVE
    elif score < -0.15:
        return SentimentLabel.NEGATIVE
    else:
        return SentimentLabel.NEUTRAL
