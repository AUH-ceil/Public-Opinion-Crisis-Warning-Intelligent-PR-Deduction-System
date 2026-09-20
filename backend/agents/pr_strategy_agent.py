# ============================================================
# Agent 4: 公关策略生成Agent — LLM + RAG 驱动版
# ============================================================
# v3.0 改造：
#   - LLM路径：充分利用RAG相似案例+品牌品类+舆情详情+竞品动作生成定制方案
#   - 降级路径：基于品牌品类+RAG案例动态构造，不再使用固定危机模板
#   - 关键区别：辣条难吃的策略 ≠ 手机爆炸的策略 ≠ 航班超售的策略
# ============================================================

import random
from typing import List, Optional, Dict, Any

from backend.models import (
    PRStrategy, RiskLevel, HistoricalCase, CompetitorAnalysis, SentimentResult
)
from backend.llm_client import call_llm_json, is_llm_available
from backend.agents.pr_templates import (
    INDUSTRY_STRATEGY_GUIDE, RISK_RESPONSE_FRAMEWORK, TALKING_POINT_GUIDELINES
)

# ==================== 策略视角随机池 ====================
# 每次 LLM 调用随机选一个视角，让多轮运行给出不同方向的建议

STRATEGY_ANGLES = [
    "本次重点关注「社交媒体传播」：如何通过微博/小红书/抖音等平台有效传递品牌态度，用什么内容形式最能打动目标消费者。",
    "本次重点关注「产品改进」：具体如何根据消费者反馈进行研发改良，包含配方/工艺/包装等层面的可执行方案。",
    "本次重点关注「消费者关系修复」：如何直接与不满意的消费者沟通，补偿方案怎么设计，如何把投诉者转化为品牌拥护者。",
    "本次重点关注「KOL/KOC合作」：如何借助第三方声音重建口碑，选择什么类型的达人，内容策略是什么。",
    "本次重点关注「差异化竞争」：如何在竞品借势的情况下保护品牌，用什么差异化优势反击，不攻击竞品但突出自身。",
    "本次重点关注「内部管理升级」：如何从组织层面（品控/客服/供应链）建立长效机制，防止类似问题再次发生。",
    "本次重点关注「品牌故事重塑」：如何通过品牌叙事把危机转化为展示企业价值观的契机，用什么故事线打动公众。",
    "本次重点关注「数据与透明度」：如何用检测报告、第三方认证、公开数据等硬证据重建消费者信任。",
    "本次重点关注「性价比和实惠」：如何通过促销/折扣/会员福利等方式让消费者感到'赚了'，用实惠对冲负面印象。",
]


# ==================== LLM System Prompt（品类感知版） ====================

PR_SYSTEM_PROMPT = """你是资深品牌公关策略顾问，服务过食品饮料、消费电子、汽车、互联网、航空酒店等多个行业。请基于当前品牌面临的舆情状况，生成有针对性的公关应对方案。

## 核心原则：
1. **策略必须匹配品类和问题类型**：
   - 食品/餐饮被说"难吃" → 研发改良+试吃活动+KOL重新评测，不是"下架召回"
   - 手机/电子质量缺陷 → 检测报告+延长保修+免费维修/换新，不是"发声明道歉"
   - 服务行业态度差 → 员工培训+流程优化+补偿，不是"CEO道歉"
   - 严重安全事故 → 才需要召回/停产/CEO出面/监管部门报告
2. **充分利用RAG历史案例**：参考相似品牌在类似问题上的成功处理经验，借鉴其话术和行动
3. **可执行**：每个步骤都是具体动作，能直接操作。禁止"高度重视""正在调查"等空洞话术
4. **分层级**：紧急(1-24h) + 短期(1-2周) + 长期(1-6个月)
5. **竞品防御**：如果竞品正在借势，需要在不攻击竞品的前提下保护品牌

## 输出格式（JSON数组，3-4条策略）：
[
  {
    "urgency": "紧急（24小时内）",
    "talking_points": ["3-5条具体话术"],
    "action_steps": ["3-6个具体动作步骤"],
    "suggested_statement": "可直接发布的声明文案"
  },
  ...
]"""


# ==================== 公关策略主函数 ====================

def pr_strategy_agent(
    brand_name: str,
    risk_level: RiskLevel,
    similar_cases: List[HistoricalCase],
    competitor_actions: List[CompetitorAnalysis],
    negative_ratio: float,
    sentiment_results: Optional[List[SentimentResult]] = None,
    brand_category: str = "",
) -> List[PRStrategy]:
    """
    【公关策略Agent v3.0】
    综合 RAG案例 + 品类特征 + 舆情详情 + 竞品动作 → 生成定制公关方案。

    参数：
        brand_name:        品牌名称
        risk_level:        风险等级
        similar_cases:     RAG 检索的相似历史案例
        competitor_actions: 竞品反应推演
        negative_ratio:    负面舆情占比
        sentiment_results: 情感分析结果（消费者具体在抱怨什么）
        brand_category:    品牌品类（如"辣条零食""新能源汽车"）

    返回：
        List[PRStrategy]: 多层级公关方案
    """
    # 从舆情分析中提取消费者核心抱怨
    complaint_keywords, complaint_tags = _extract_complaints(sentiment_results or [])
    # 品类代理（用于降级）
    category = brand_category or _infer_category_from_cases(similar_cases)

    # === 尝试 LLM 生成 ===
    if is_llm_available():
        try:
            results = _pr_with_llm(
                brand_name=brand_name,
                risk_level=risk_level,
                category=category,
                similar_cases=similar_cases,
                competitor_actions=competitor_actions,
                negative_ratio=negative_ratio,
                complaint_keywords=complaint_keywords,
                complaint_tags=complaint_tags,
            )
            if results:
                print(f"[公关策略Agent] LLM生成：{len(results)} 套方案（品类={category}）")
                return results
        except Exception as e:
            print(f"[公关策略Agent] LLM失败: {e}，降级动态构造")

    # === 降级：基于品类 + RAG案例动态构造 ===
    fallback = _pr_dynamic_fallback(
        brand_name=brand_name,
        risk_level=risk_level,
        category=category,
        similar_cases=similar_cases,
        competitor_actions=competitor_actions,
        complaint_tags=complaint_tags,
    )
    print(f"[公关策略Agent] 动态降级：{len(fallback)} 套方案")
    return fallback


# ==================== LLM 模式 ====================

def _pr_with_llm(
    brand_name: str,
    risk_level: RiskLevel,
    category: str,
    similar_cases: List[HistoricalCase],
    competitor_actions: List[CompetitorAnalysis],
    negative_ratio: float,
    complaint_keywords: List[str],
    complaint_tags: List[str],
) -> List[PRStrategy]:
    """用 LLM 生成品类定制的公关方案"""

    # 构建RAG案例上下文（这是LLM最重要的参考）
    cases_context = ""
    if similar_cases:
        cases_context = "## 历史相似案例及处理方案（RAG检索，请重点参考）：\n" + "\n".join([
            f"【案例{i+1}】{c.title}\n"
            f"  分类：{c.category} | 风险等级：{c.risk_level.value}\n"
            f"  处理方案：{c.resolution}\n"
            for i, c in enumerate(similar_cases[:5])
        ])

    # 竞品威胁
    competitor_context = ""
    if competitor_actions:
        competitor_context = "## 竞品正在做什么：\n" + "\n".join([
            f"- {c.competitor_name}：{c.action_type}（{c.probability*100:.0f}%）— {c.description}"
            for c in competitor_actions[:5]
        ])

    # 响应级别
    response_urgency = {
        RiskLevel.CRITICAL: "危机级别——需要CEO出面，1小时内发布官方声明",
        RiskLevel.HIGH: "高级别——品牌负责人出面，2小时内回应",
        RiskLevel.MEDIUM: "中等级别——24小时内通过官方渠道说明情况",
        RiskLevel.LOW: "低级别——持续监控，适时回应，不需要过度反应",
    }.get(risk_level, "常规响应")

    # 随机选择一个策略视角（让多次运行给出不同方向的建议）
    angle = random.choice(STRATEGY_ANGLES)

    user_prompt = f"""
## 品牌与问题概况
- 品牌：{brand_name}
- 品类：{category}
- 风险等级：{risk_level.value}（{response_urgency}）
- 负面舆情占比：{negative_ratio*100:.0f}%
- 消费者核心抱怨词：{', '.join(complaint_keywords) if complaint_keywords else '暂无详情'}
- 问题标签：{', '.join(complaint_tags) if complaint_tags else '暂无分类'}

{cases_context}

{competitor_context}

## >> 本次策略侧重
{angle}

---
请基于以上信息，为{brand_name}生成3套分层公关应对方案（紧急→短期→长期）。
在「{angle[:20]}...」这个方向上给出更多具体、可落地的建议。
方案必须匹配「{category}」品类的特性——不要生搬硬套其他行业的做法。
声明文案要可以直接发布，包含品牌态度、具体行动、消费者补偿（如适用）。"""

    parsed = call_llm_json(
        system_prompt=PR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.8,  # 适度提高温度，增加输出多样性
        max_tokens=4096,
    )

    if "error" in parsed:
        return []

    strategies_data = parsed if isinstance(parsed, list) else parsed.get("strategies", [])
    if not isinstance(strategies_data, list):
        return []

    results = []
    for s in strategies_data[:5]:
        if not isinstance(s, dict):
            continue
        results.append(PRStrategy(
            risk_level=risk_level,
            urgency=str(s.get("urgency", "常规")),
            talking_points=list(s.get("talking_points", []))[:8],
            action_steps=list(s.get("action_steps", []))[:8],
            suggested_statement=str(s.get("suggested_statement", "")),
        ))
    return results


# ==================== 动态降级策略（不用固定模板） ====================

def _pr_dynamic_fallback(
    brand_name: str,
    risk_level: RiskLevel,
    category: str,
    similar_cases: List[HistoricalCase],
    competitor_actions: List[CompetitorAnalysis],
    complaint_tags: List[str],
) -> List[PRStrategy]:
    """
    基于品牌品类 + RAG案例动态生成策略（LLM 不可用时的降级方案）。

    原理：
      - 如果 RAG 搜到了相似案例 → 直接参考其处理方案
      - 如果没搜到 → 根据品类和风险等级生成通用但合理的策略
      - 不再使用 pr_templates.py 的"CEO道歉/产品下架/全球召回"模板
    """
    strategies = []

    # === Step 1: 如果有 RAG 案例，直接引用其方案 ===
    if similar_cases:
        for i, case in enumerate(similar_cases[:3]):
            resolution_steps = _parse_resolution_steps(case.resolution)
            strategies.append(PRStrategy(
                risk_level=risk_level,
                urgency=f"参考案例：{case.title}" if i == 0 else f"备选方案{i+1}",
                talking_points=[
                    f"参考「{case.title}」的成功处理经验",
                    f"该案例为{case.category}类问题，与本品牌情况相近",
                    f"核心思路：{resolution_steps[0] if resolution_steps else case.resolution[:80]}",
                ],
                action_steps=resolution_steps[:6] if resolution_steps else [
                    f"① 参照案例「{case.title}」的处理流程执行",
                    f"② 根据{brand_name}实际情况调整具体措施",
                    f"③ 持续监测舆情变化，评估效果",
                ],
                suggested_statement="",
            ))
        # 加上竞品防御
        if competitor_actions:
            strategies.append(_build_competitor_defense(competitor_actions[0], risk_level))
        return strategies

    # === Step 2: 没有RAG案例，按品类+风险等级生成 ===
    severity = _severity_label(risk_level)

    # 品类定制策略
    if any(kw in category for kw in ["食品", "零食", "辣条", "饮料", "饮品", "餐饮", "奶茶", "咖啡"]):
        strategies = _food_strategies(brand_name, severity, complaint_tags)
    elif any(kw in category for kw in ["手机", "电子", "汽车", "新能源", "数码", "家电"]):
        strategies = _product_strategies(brand_name, severity, complaint_tags)
    elif any(kw in category for kw in ["酒店", "航空", "旅游", "服务", "快递", "外卖"]):
        strategies = _service_strategies(brand_name, severity, complaint_tags)
    else:
        strategies = _generic_strategies(brand_name, severity, category, complaint_tags)

    # 竞品防御
    if competitor_actions:
        strategies.append(_build_competitor_defense(competitor_actions[0], risk_level))

    return strategies


# ==================== 品类定制策略 ====================

def _food_strategies(brand: str, severity: str, tags: List[str]) -> List[PRStrategy]:
    """食品/餐饮品牌的公关策略"""
    is_crisis = severity in ("特级风险", "高风险")

    if is_crisis:
        return [
            PRStrategy(risk_level=RiskLevel.HIGH, urgency="紧急（24小时内）",
                talking_points=[
                    f"承认{brand}在产品上确实存在问题，不推卸责任",
                    "强调食品安全是品牌生命线，对消费者健康负责",
                    "公布已采取的具体措施和时间表",
                ],
                action_steps=[
                    "① 立即下架/停售受质疑的产品批次，发布召回公告",
                    "② 送检第三方权威机构，48小时内公布检测报告",
                    "③ 开通消费者退款/赔偿绿色通道",
                    "④ 品牌负责人在官方渠道出面回应",
                    "⑤ 配合监管部门检查，主动报告",
                ],
                suggested_statement="",
            ),
            PRStrategy(risk_level=RiskLevel.HIGH, urgency="长期（1-3个月品牌修复）",
                talking_points=[
                    "引入国际食品安全认证体系",
                    "建立透明化生产流程，邀请消费者监督",
                    "定期发布品质报告，重建消费者信任",
                ],
                action_steps=[
                    "① 升级品控标准，引入HACCP/ISO22000认证",
                    "② 每月发布食品安全报告，公开检测数据",
                    "③ 开展工厂开放日活动，邀请消费者/KOL参观",
                    "④ 建立消费者监督委员会",
                    "⑤ 研发改良配方，提升产品品质",
                ],
                suggested_statement="",
            ),
        ]
    else:
        return [
            PRStrategy(risk_level=RiskLevel.LOW, urgency="日常运营优化",
                talking_points=[
                    "认真听取消费者反馈，口味问题因人而异",
                    f"{brand}一直在优化产品配方和工艺",
                    "欢迎消费者通过官方渠道提出建议",
                ],
                action_steps=[
                    "① 整理消费者反馈中的共性问题，提交研发部门",
                    "② 针对高频差评维度（口味/口感/包装等）进行改良测试",
                    "③ 小范围推出改良版，收集种子用户反馈",
                    "④ 邀请美食KOL重新评测改良产品",
                    "⑤ 在社交媒体开展试吃/试用活动，积累正面评价",
                ],
                suggested_statement="",
            ),
            PRStrategy(risk_level=RiskLevel.LOW, urgency="消费者互动与口碑维护",
                talking_points=[
                    "真诚回复每一条差评，展示品牌负责任的态度",
                    "对不满意的消费者提供补偿（优惠券/退款/换货）",
                    "用真实的消费者好评自然覆盖负面声音",
                ],
                action_steps=[
                    "① 在电商平台/社交媒体逐条回复消费者差评",
                    "② 建立消费者反馈→研发的快速响应机制",
                    "③ 开展'产品体验官'招募活动，让真实用户为品牌发声",
                ],
                suggested_statement="",
            ),
        ]


def _product_strategies(brand: str, severity: str, tags: List[str]) -> List[PRStrategy]:
    """电子/汽车等耐用消费品策略"""
    return [
        PRStrategy(risk_level=RiskLevel.MEDIUM, urgency="品质回应",
            talking_points=[
                "拿出数据和检测报告说话",
                "明确区分个案和普遍问题",
                "展示完善的售后服务体系",
            ],
            action_steps=[
                f"① 针对消费者反馈的{'/'.join(tags) if tags else '问题'}进行技术检测",
                "② 发布检测报告或技术说明",
                "③ 延长相关部件的保修期",
                "④ 升级品控流程，公布改进措施",
            ],
            suggested_statement="",
        ),
        PRStrategy(risk_level=RiskLevel.MEDIUM, urgency="用户关系维护",
            talking_points=["每一位用户的反馈都是我们进步的动力", "用实际行动证明品质承诺"],
            action_steps=[
                "① 主动联系反馈问题的用户，提供免费检测/维修",
                "② 设立用户反馈专项基金",
                "③ 每季度发布品质白皮书",
            ],
            suggested_statement="",
        ),
    ]


def _service_strategies(brand: str, severity: str, tags: List[str]) -> List[PRStrategy]:
    """酒店/航空/外卖等服务行业策略"""
    return [
        PRStrategy(risk_level=RiskLevel.MEDIUM, urgency="服务补救",
            talking_points=[
                "承认服务中的不足，向消费者致歉",
                "公布具体的服务改进措施",
                "展示品牌对服务质量的重视",
            ],
            action_steps=[
                "① 核实消费者反馈的具体服务事件",
                "② 向受影响消费者致歉并提供补偿",
                "③ 涉事员工/门店停职培训",
                "④ 升级服务标准和考核机制",
            ],
            suggested_statement="",
        ),
        PRStrategy(risk_level=RiskLevel.MEDIUM, urgency="长期服务升级",
            talking_points=["将消费者投诉视为服务升级的契机", "建立行业领先的服务标准"],
            action_steps=[
                "① 全面服务流程审查和优化",
                "② 全员服务意识和技能培训",
                "③ 引入消费者满意度实时评价系统",
                "④ 设立服务质量监督热线",
            ],
            suggested_statement="",
        ),
    ]


def _generic_strategies(brand: str, severity: str, category: str,
                        tags: List[str]) -> List[PRStrategy]:
    """
    通用策略 — 先尝试从 pr_templates 知识库匹配行业，
    匹配不到再用最通用的框架。
    """
    # === 尝试从知识库匹配行业 ===
    for industry, guide in INDUSTRY_STRATEGY_GUIDE.items():
        # 品类关键词与行业名匹配
        if any(kw in category for kw in [industry] + industry.split("/")):
            # 找到匹配行业，尝试按问题类型匹配
            common_issues = guide.get("常见问题", [])
            matched_issue = None
            for issue in common_issues:
                if any(tag in issue for tag in tags):
                    matched_issue = issue
                    break
            # 没匹配到具体问题就用第一个
            if not matched_issue and common_issues:
                matched_issue = common_issues[0]

            if matched_issue and matched_issue in guide:
                recipe = guide[matched_issue]
                strategies = []
                # 紧急策略
                urgent_steps = recipe.get("紧急(24小时内)", recipe.get("紧急(1-3天)", recipe.get("紧急", [])))
                strategies.append(PRStrategy(
                    risk_level=RiskLevel.LOW,
                    urgency="紧急应对",
                    talking_points=[f"针对{brand}的{matched_issue}问题", recipe.get("思路", "")[:120]],
                    action_steps=[f"{i+1}⃞ {s}" for i, s in enumerate(urgent_steps[:6])],
                    suggested_statement="",
                ))
                # 短期策略
                short_steps = recipe.get("短期(1-4周)", recipe.get("短期(1-2周)", recipe.get("短期", [])))
                if short_steps:
                    strategies.append(PRStrategy(
                        risk_level=RiskLevel.LOW,
                        urgency="短期优化",
                        talking_points=["分阶段推进改进措施"],
                        action_steps=[f"{i+1}⃞ {s}" for i, s in enumerate(short_steps[:6])],
                        suggested_statement="",
                    ))
                # 长期策略
                long_steps = recipe.get("长期(1-3月)", recipe.get("长期(1-6月)", recipe.get("长期", [])))
                if long_steps:
                    strategies.append(PRStrategy(
                        risk_level=RiskLevel.LOW,
                        urgency="长期建设",
                        talking_points=["建立长效机制"],
                        action_steps=[f"{i+1}⃞ {s}" for i, s in enumerate(long_steps[:6])],
                        suggested_statement="",
                    ))
                if strategies:
                    return strategies

    # === 知识库也没匹配到，用最通用的框架 ===
    return [
        PRStrategy(risk_level=RiskLevel.LOW, urgency="舆情回应",
            talking_points=[f"关注消费者对{brand}的反馈", "以开放态度面对批评"],
            action_steps=[
                f"① 收集和分析消费者反馈中的核心问题",
                f"② 针对具体问题进行内部核查和改进",
                f"③ 通过官方渠道与消费者沟通进展",
            ],
            suggested_statement="",
        ),
        PRStrategy(risk_level=RiskLevel.LOW, urgency="品牌建设",
            talking_points=["将反馈转化为品牌进步的动力"],
            action_steps=[
                "① 建立消费者反馈闭环机制",
                "② 定期公开改进成果",
                "③ 加强与消费者的日常互动",
            ],
            suggested_statement="",
        ),
    ]


def _build_competitor_defense(top_competitor: CompetitorAnalysis,
                              risk_level: RiskLevel) -> PRStrategy:
    """生成竞品防御策略"""
    return PRStrategy(
        risk_level=risk_level,
        urgency="竞品防御（同步执行）",
        talking_points=[
            f"密切关注{top_competitor.competitor_name}的{top_competitor.action_type}动向",
            "准备品牌优势和第三方认证材料，用事实说话",
            "加强社交媒体正面内容产出，自然覆盖负面声音",
        ],
        action_steps=[
            "① 启动竞品舆情监控预警",
            "② 准备品牌差异化对比材料（品质/价格/服务）",
            "③ 加强KOL/KOC合作关系，稳定品牌声量",
            "④ 如遇恶意抹黑，通过法律途径维权",
        ],
        suggested_statement="",
    )


# ==================== 辅助函数 ====================

def _extract_complaints(sentiment_results: List[SentimentResult]
                        ) -> tuple[List[str], List[str]]:
    """从情感分析结果中提取消费者核心抱怨词和问题标签"""
    kw_counter: Dict[str, int] = {}
    tag_counter: Dict[str, int] = {}

    for r in sentiment_results:
        if hasattr(r, 'negative_keywords') and r.negative_keywords:
            for kw in r.negative_keywords:
                kw_counter[kw] = kw_counter.get(kw, 0) + 1
        if hasattr(r, 'risk_tags') and r.risk_tags:
            for tag in r.risk_tags:
                tag_counter[tag] = tag_counter.get(tag, 0) + 1

    top_kw = sorted(kw_counter.items(), key=lambda x: x[1], reverse=True)
    top_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)

    return [kw for kw, _ in top_kw[:10]], [tag for tag, _ in top_tags[:5]]


def _infer_category_from_cases(similar_cases: List[HistoricalCase]) -> str:
    """从RAG案例推断品牌品类"""
    if similar_cases:
        cat = similar_cases[0].category
        return cat if cat else "消费品"
    return "消费品"


def _parse_resolution_steps(resolution: str) -> List[str]:
    """解析RAG案例中的处理方案，提取步骤"""
    if not resolution:
        return []
    # 按常见分隔符拆分
    steps = []
    # 先尝试按序号拆分
    import re
    numbered = re.split(r'(?:第[一二三四五六七八九十\d]+步[：:]|[\d]+[\.\、\)）])', resolution)
    if len(numbered) > 1:
        steps = [s.strip() for s in numbered[1:] if s.strip()]
    if not steps:
        # 按句号分
        steps = [s.strip() + "。" for s in resolution.split("。") if s.strip()][:5]
    return steps


def _severity_label(risk_level: RiskLevel) -> str:
    mapping = {
        RiskLevel.CRITICAL: "特级风险",
        RiskLevel.HIGH: "高风险",
        RiskLevel.MEDIUM: "中风险",
        RiskLevel.LOW: "低风险",
    }
    return mapping.get(risk_level, "中风险")
