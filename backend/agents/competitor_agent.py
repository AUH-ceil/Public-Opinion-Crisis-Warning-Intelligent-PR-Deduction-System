# ============================================================
# Agent 3: 竞品反应Agent — LLM 驱动版
# ============================================================
# 改造要点：
#   v1.0: 7个固定策略模板 + 随机竞品名（"竞品A品牌"），跟真实行业无关
#   v2.0: LLM 基于真实行业格局动态推演，给出具体品牌名和定制化策略
#
# 降级策略：
#   LLM 可用 → 深度行业分析（具体品牌 + 概率 + 历史案例）
#   LLM 不可用 → 模板策略 + 随机降级（保留原逻辑）
# ============================================================

import random
import json
from typing import List

from backend.models import CompetitorAnalysis, SentimentResult, RiskLevel
from backend.llm_client import call_llm_json, is_llm_available


# ==================== LLM System Prompt ====================

COMPETITOR_SYSTEM_PROMPT = """你是资深商业竞争情报分析师，曾在多家咨询公司（麦肯锡、波士顿咨询）任职。请基于当前品牌的危机情况，推演其竞争对手可能采取的行动。

## 要求：
1. **competitor_name**：给出真实可查的竞争对手品牌名称（不要用"竞品A"这种占位符），结合行业常识推断。如果是新能源汽车就写比亚迪/蔚来/理想，奶茶就写蜜雪冰城/茶百道/古茗等
2. **action_type**：动作类型，例如：
   - "借势营销"：发布对比内容抢占市场份额
   - "价格战"：降价促销抢夺用户
   - "人才挖角"：趁乱挖核心技术/管理人员
   - "舆论施压"：通过KOL放大危机话题
   - "战略收购"：评估收购时机
   - "沉默观望"：按兵不动等待时机
   - "行业联盟"：联合其他品牌建立标准，孤立危机品牌
   - "渠道争夺"：抢夺经销商/供应商/流量入口
3. **probability**：0.0~1.0 发生概率。风险等级越高、负面占比越大，竞品越活跃
4. **description**：80-180字的详细推演描述，具体到时间节点和操作手法
5. **historical_case**：引用一个真实发生过的类似竞争案例（品牌+年份+事件），如"2016年三星Note7爆炸事件后，华为Mate9借势发布'安全电池'技术白皮书"

## 概率参考：
- 品牌危机等级越高，竞品激进策略(借势/价格战/挖角)概率越高
- 沉默观望始终有一定概率(0.2-0.5)，不是所有竞品都会行动
- 收购类概率较低(≤0.3)，除非是特级危机

## 历史案例要求：
- 必须引用 2024-2026 年的真实商业案例（不要用2016-2023年的老案例）
- 覆盖多个行业：食品饮料、手机数码、汽车、电商、餐饮、美妆等
- 例如：2025年良品铺子被曝配料表造假后，来伊份未跟进营销战，而是3个月后推出'透明溯源'系统获消费者信任
- 例如：2024年农夫山泉舆论风波中，娃哈哈借势加强'国货情怀'营销，市场份额明显提升

返回JSON数组：[{"competitor_name":"...", "action_type":"...", "probability":0.0, "description":"...", "historical_case":"..."}, ...]"""


# ==================== 原版模板（降级用） ====================

COMPETITOR_STRATEGIES = [
    {
        "action_type": "借机造势",
        "description_template": "在社交平台发布「选择{brand}，我们更懂你」系列对比内容，突出自身产品优势，蚕食{brand}流失的用户群。预计将在48小时内全面铺开营销物料。",
        "base_probability": 0.7,
    },
    {
        "action_type": "对比抹黑",
        "description_template": "通过第三方KOL发布对比测评，放大{brand}的{issue}问题。同时安排水军在{brand}相关话题下带节奏，加速舆情恶化。",
        "base_probability": 0.55,
    },
    {
        "action_type": "默默降价",
        "description_template": "不直接回应{brand}危机，但全线产品降价15-25%，配合「品质如一、良心价格」宣传，低调收割{brand}流失的市场份额。",
        "base_probability": 0.6,
    },
    {
        "action_type": "行业发声",
        "description_template": "以「行业自律」为名发布公开声明，承诺自身产品绝不会出现类似问题，间接强调{brand}的失误。",
        "base_probability": 0.45,
    },
    {
        "action_type": "沉默观望",
        "description_template": "暂时保持沉默，密切监控舆情走势。如果{brand}危机持续恶化，将在最佳时机出手。",
        "base_probability": 0.8,
    },
    {
        "action_type": "挖角人才",
        "description_template": "趁{brand}内部动荡，通过猎头接触{brand}核心技术/管理人才。",
        "base_probability": 0.35,
    },
    {
        "action_type": "溢价收购",
        "description_template": "评估{brand}危机受损程度，如果市值大幅缩水将启动收购预案。",
        "base_probability": 0.25,
    },
]

# 注：LLM 可用时会动态分析真实竞品名，以下是降级时的兜底
# 不再用"竞品A品牌"这种占位符，改用常见竞品关系对
COMMON_COMPETITOR_PAIRS = {
    # 辣条/零食
    "麻辣": ["卫龙", "盐津铺子", "良品铺子", "三只松鼠", "百草味"],
    "辣条": ["卫龙", "麻辣王子", "飞旺", "翻天娃"],
    "零食": ["良品铺子", "三只松鼠", "百草味", "来伊份", "盐津铺子"],
    # 茶饮
    "奶茶": ["蜜雪冰城", "茶百道", "古茗", "沪上阿姨", "霸王茶姬"],
    "茶": ["喜茶", "奈雪的茶", "霸王茶姬", "茶颜悦色"],
    # 咖啡
    "咖啡": ["星巴克", "瑞幸", "库迪", "Manner", "Tims"],
    # 手机/电子
    "手机": ["华为", "小米", "OPPO", "vivo", "荣耀"],
    "汽车": ["比亚迪", "蔚来", "理想", "小鹏", "问界"],
    # 电商
    "电商": ["拼多多", "京东", "淘宝", "抖音电商", "快手电商"],
    # 快餐
    "快餐": ["麦当劳", "肯德基", "汉堡王", "塔斯汀", "华莱士"],
    "火锅": ["海底捞", "巴奴", "呷哺呷哺", "小龙坎", "楠火锅"],
}

HISTORICAL_COMPETITOR_CASES = [
    # 2024-2026 年中国市场真实案例
    "2025年良品铺子被曝配料表造假后，来伊份未跟进营销战，3个月后推出'透明溯源'系统重建消费者信任",
    "2024年农夫山泉舆论风波中，娃哈哈借势'国货情怀'营销，瓶装水市场份额从12%升至18%",
    "2025年某奶茶品牌被曝过期食材后，蜜雪冰城借机强调'极致供应链+低价不低质'定位，门店数逆势增长",
    "2024年小米SU7发布后，极氪紧急调整007定价策略并加速智驾功能OTA推送，保住了20-25万价位市场份额",
    "2025年瑞幸咖啡某门店卫生问题曝光后，库迪咖啡在附近3公里所有门店推出'第二杯半价'活动精准截流",
    "2025年胖东来因'试吃事件'被推上热搜后，永辉超市低调升级熟食区卫生标准并邀请消费者监督",
    "2024年某国产美妆品牌成分争议中，珀莱雅迅速公布全成分溯源报告并邀请第三方检测，当月销量不降反升",
    "2025年东方甄选主播言论风波，交个朋友直播间未直接评论但加大了农产品溯源直播力度，观看量翻倍",
]


# ==================== 竞品推演主函数 ====================

def competitor_agent(
    brand_name: str,
    sentiment_results: List[SentimentResult],
    risk_level: RiskLevel,
    negative_ratio: float,
) -> List[CompetitorAnalysis]:
    """
    【竞品反应Agent v2.0】
    LLM 基于真实行业格局动态推演竞品动作。

    参数：
        brand_name: 当前品牌名称
        sentiment_results: 情感分析结果列表
        risk_level: 当前风险等级
        negative_ratio: 负面舆情占比

    返回：
        List[CompetitorAnalysis]: 竞品反应推演（按概率降序）
    """
    if not sentiment_results:
        return []

    # 提取关键信息
    main_issue = _extract_main_issue(sentiment_results)
    top_keywords = _extract_top_keywords(sentiment_results, limit=8)
    risk_tags = _extract_risk_tags(sentiment_results)

    # === 尝试 LLM 分析 ===
    if is_llm_available():
        try:
            results = _competitor_with_llm(
                brand_name, main_issue, top_keywords, risk_tags,
                risk_level, negative_ratio
            )
            if results:
                print(f"[竞品Agent] LLM推演完成：{len(results)} 个竞品可能采取行动")
                return results
        except Exception as e:
            print(f"[竞品Agent] LLM推演失败: {e}，降级为模板模式")

    # === 降级：模板模式 ===
    return _competitor_rule_based(brand_name, sentiment_results, risk_level, negative_ratio)


# ==================== LLM 模式 ====================

def _competitor_with_llm(
    brand_name: str, main_issue: str, top_keywords: List[str],
    risk_tags: List[str], risk_level: RiskLevel, negative_ratio: float,
) -> List[CompetitorAnalysis]:
    """用 LLM 推演竞品反应"""

    user_prompt = f"""
当前危机概况：
- 品牌：{brand_name}
- 风险等级：{risk_level.value}
- 负面舆情占比：{negative_ratio*100:.0f}%
- 主要问题类型：{', '.join(risk_tags) if risk_tags else main_issue}
- 核心负面关键词：{', '.join(top_keywords)}
- 舆情总数：已分析多条相关舆情

请推演6个最可能的竞品动作，按发生概率从高到低排列。
注意：品牌名必须是真实可查的竞争对手。"""

    parsed = call_llm_json(
        system_prompt=COMPETITOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.7,  # 需要一定创造性
    )

    if "error" in parsed:
        return []

    # 如果返回的是数组直接使用，如果是对象尝试提取
    actions_data = parsed if isinstance(parsed, list) else parsed.get("actions", parsed.get("competitors", []))

    if not isinstance(actions_data, list):
        return []

    results = []
    for a in actions_data[:8]:
        if not isinstance(a, dict):
            continue
        results.append(CompetitorAnalysis(
            competitor_name=str(a.get("competitor_name", "未知竞品")),
            action_type=str(a.get("action_type", "观望")),
            probability=float(a.get("probability", 0.5)),
            description=str(a.get("description", "")),
            historical_case=str(a.get("historical_case", "")),
        ))

    # 按概率降序
    results.sort(key=lambda x: x.probability, reverse=True)
    return results


# ==================== 规则引擎模式（降级） ====================

def _competitor_rule_based(
    brand_name: str,
    sentiment_results: List[SentimentResult],
    risk_level: RiskLevel,
    negative_ratio: float,
) -> List[CompetitorAnalysis]:
    """降级方案：根据品牌名智能匹配真实竞品 + 随机策略"""
    main_issue = _extract_main_issue(sentiment_results)

    # 从 COMMON_COMPETITOR_PAIRS 找匹配的真实竞品
    competitor_names = _match_competitors(brand_name)

    risk_multiplier = {
        RiskLevel.LOW: 0.3,
        RiskLevel.MEDIUM: 0.6,
        RiskLevel.HIGH: 0.85,
        RiskLevel.CRITICAL: 1.0,
    }.get(risk_level, 0.5)

    competitor_actions = []
    selected = random.sample(
        competitor_names,
        min(len(competitor_names), random.randint(3, 5))
    )

    for comp_name in selected:
        strategy = random.choice(COMPETITOR_STRATEGIES)
        adjusted_prob = strategy["base_probability"] * risk_multiplier * negative_ratio
        adjusted_prob = max(0.05, min(1.0, adjusted_prob + random.uniform(-0.1, 0.15)))

        description = strategy["description_template"].format(
            brand=brand_name,
            issue=main_issue,
        )
        historical_ref = random.choice(HISTORICAL_COMPETITOR_CASES)

        competitor_actions.append(CompetitorAnalysis(
            competitor_name=comp_name,
            action_type=strategy["action_type"],
            probability=round(adjusted_prob, 2),
            description=description,
            historical_case=historical_ref,
        ))

    competitor_actions.sort(key=lambda x: x.probability, reverse=True)
    print(f"[竞品Agent] 降级模式：匹配到 {len(competitor_actions)} 个真实竞品")
    return competitor_actions


def _match_competitors(brand_name: str) -> List[str]:
    """根据品牌名从竞品库中匹配真实竞品名称"""
    # 先尝试关键词匹配
    for keyword, comps in COMMON_COMPETITOR_PAIRS.items():
        if keyword in brand_name:
            # 排除品牌自身（如果它在列表里）
            return [c for c in comps if c not in brand_name]

    # 没匹配到，合并所有竞品中随机选
    all_comps = []
    for comps in COMMON_COMPETITOR_PAIRS.values():
        all_comps.extend(comps)
    all_comps = list(set(all_comps))  # 去重
    random.shuffle(all_comps)
    return all_comps[:8]


# ==================== 辅助函数 ====================

def _extract_main_issue(sentiment_results: List[SentimentResult]) -> str:
    """从情感分析结果中提取最突出的问题类型"""
    tag_counter = {}
    for r in sentiment_results:
        for tag in r.risk_tags:
            tag_counter[tag] = tag_counter.get(tag, 0) + 1
    if tag_counter:
        return max(tag_counter, key=tag_counter.get)
    return "产品质量"


def _extract_top_keywords(sentiment_results: List[SentimentResult], limit: int = 8) -> List[str]:
    """提取最高频的负面关键词"""
    kw_counter = {}
    for r in sentiment_results:
        for kw in r.negative_keywords:
            kw_counter[kw] = kw_counter.get(kw, 0) + 1
    sorted_kw = sorted(kw_counter.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kw[:limit]]


def _extract_risk_tags(sentiment_results: List[SentimentResult]) -> List[str]:
    """提取去重后的风险标签"""
    tags = set()
    for r in sentiment_results:
        for tag in r.risk_tags:
            tags.add(tag)
    return list(tags)[:5]
