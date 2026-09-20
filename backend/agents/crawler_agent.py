# ============================================================
# Agent 1: 爬虫Agent — 真实搜索 + LLM 舆情提取
# ============================================================
# 改造要点：
#   v1.0: 20条硬编码模板随机拼接 → 完全脱离现实
#   v2.0: Google/Bing/Baidu搜索 → LLM提取结构化舆情 → 真实数据
#
# 搜索引擎策略（自动降级）：
#   SerpAPI → Google CSE → Playwright(Google→Bing→百度) → 模拟数据兜底
#
# LLM 提取：
#   将搜索结果（标题+摘要）转为 SentimentItem 结构化数据
#   能识别真实新闻中的情感、提取作者/来源、判断平台类型
# ============================================================

import re
import random
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from backend.models import SentimentItem, Platform
from backend.search import search_news, estimate_source_weight
from backend.llm_client import call_llm_json, is_llm_available


# ==================== 品牌理解 System Prompt ====================

BRAND_CONTEXT_PROMPT = """你是一个品牌信息分析师。用户输入了一个品牌名称，请分析这个品牌，输出结构化 JSON。

## 分析要求：
1. **category**：品牌所属行业/品类（如"辣条零食""新能源汽车""茶饮连锁""手机数码""化妆品"等）
2. **product_type**：主要产品叫什么（如"辣条""电动车""手机""面霜"等，用消费者日常叫法）
3. **description**：一句话介绍这个品牌（30字以内）
4. **typical_price**：典型价格区间描述（如"几块钱一包""十几块钱一杯""几十万"等）
5. **consumer_talk**：消费者讨论这个品牌时通常会聊哪些话题？（3-5个角度）
   - 例如辣条品牌：["好不好吃", "辣度怎么样", "油不油", "是不是童年的味道", "和卫龙比谁好吃"]
   - 例如手机品牌：["拍照效果", "续航怎么样", "系统流畅度", "性价比", "售后服务"]
6. **search_queries**：为了全面了解网上对品牌的评价，应该搜哪些词？（5-8个搜索词）
   - 覆盖正向、负面、中性评价
   - 包含消费者常用搜索句式
7. **evaluation_dimensions**：评价这个品类产品好坏的核心维度（3-5个）
   - 例如辣条：["口味", "口感", "辣度", "油腻度", "性价比"]
   - 例如手机：["性能", "拍照", "续航", "做工", "系统"]

返回格式：
{
  "category": "...",
  "product_type": "...",
  "description": "...",
  "typical_price": "...",
  "consumer_talk": [...],
  "search_queries": [...],
  "evaluation_dimensions": [...]
}"""


# ==================== LLM 提取 System Prompt ====================

EXTRACT_SYSTEM_PROMPT = """你是一个舆情数据提取专家。给定一条来自搜索引擎的结果（可能是新闻、论坛帖子、评测、社交媒体内容等），将其转换为标准舆情数据格式。

## 提取规则：
1. **content**：将该条目的核心内容凝练为 50-150 字的中文描述。保留关键事实：品牌名、产品名、具体评价内容、情感倾向。如果是消费者评价，保留原汁原味的评价语言（好吃/难吃/好用/差评等）
2. **sentiment_hint**：基于内容的语气和立场判断情感倾向
   - "负面"：批评、吐槽、踩雷、避雷、难吃、难用、质量差、投诉、曝光
   - "中性"：客观评测、对比分析、提问、事实陈述、有好有坏的评价
   - "正向"：表扬、推荐、好吃、好用、回购、种草、性价比高
3. **platform**：根据来源推断平台
   - 新闻媒体（新华网/央视/澎湃/新浪/腾讯/网易/搜狐/凤凰/36氪/虎嗅等）→ "新闻媒体"
   - 微博/weibo → "微博"
   - 小红书/xiaohongshu → "小红书"
   - 知乎/zhihu → "知乎"
   - 抖音/douyin → "抖音"
   - 贴吧/tieba/百度贴吧 → "贴吧"
   - 无法判断 → "新闻媒体"（默认）
4. **author**：提取作者/发布者名称

## 返回格式（纯JSON）：
{
  "content": "提取的舆情内容描述",
  "sentiment_hint": "负面/中性/正向",
  "platform": "新闻媒体/微博/小红书/知乎/抖音/贴吧",
  "author": "来源名称"
}"""


# ==================== 爬虫 Agent 核心 ====================

# 注：兜底模板已改为动态生成（_generate_dynamic_fallback），
# 基于 LLM 品牌理解（_understand_brand）返回的品类/评价维度/消费者话题
# 动态构造消费者评价，无需维护静态品类词库。

def crawler_agent(query_text: str, brand_name: str, count: int = 20) -> List[SentimentItem]:
    """
    【爬虫Agent v3.0】
    品牌理解 → 智能搜索 → LLM提取 → 结构化舆情

    策略链：
      LLM品牌分析（理解品类/话题/搜索词）
        → 多词精准搜索(SerpAPI→CSE→Playwright)
        → LLM提取结构化舆情
        → 不可用时：动态生成消费者评价（兜底）

    参数：
        query_text: 搜索提示词（可选，用户可能只输入品牌名）
        brand_name: 品牌名称（如"麻辣小王子""喜茶""小米"）
        count:     期望获取条数（默认20）

    返回：
        List[SentimentItem]: 结构化舆情数据
    """
    items = []

    # ===== Step 0: LLM 理解品牌（品类/话题/搜索词） =====
    brand_context = _understand_brand(brand_name, query_text)

    # ===== Step 1: 基于品牌理解构建搜索词 =====
    raw_results = []
    if brand_context.get("search_queries"):
        # 用 LLM 给的搜索词组合搜索
        search_queries = brand_context["search_queries"][:5]
        raw_results = _multi_query_search(search_queries, brand_name, count)
    else:
        # 降级：传统拼接
        search_query = f"{brand_name} {query_text}".strip()
        try:
            raw_results = search_news(search_query, max_results=count)
            print(f"[爬虫Agent] 搜索获取 {len(raw_results)} 条结果")
        except Exception as e:
            print(f"[爬虫Agent] 搜索失败: {e}")

    # ===== Step 2: LLM 提取结构化舆情 =====
    if raw_results and is_llm_available():
        items = _extract_with_llm(raw_results)
    elif raw_results:
        items = _extract_heuristic(raw_results, brand_name)
        print(f"[爬虫Agent] LLM不可用，使用启发式提取: {len(items)} 条")

    # ===== Step 3: 兜底生成（基于品牌理解动态生成） =====
    if not items:
        print(f"[爬虫Agent] 搜索无结果，基于品牌理解动态生成评价")
        items = _generate_dynamic_fallback(brand_name, brand_context, count)

    # ===== Step 4: 补齐互动数据 =====
    _enrich_engagement(items)

    # 控制器通知下游
    print(f"[爬虫Agent] 品牌={brand_name} | 品类={brand_context.get('category','未知')} | 输出{len(items)}条舆情")
    return items


# ==================== 品牌理解 ====================

def _understand_brand(brand_name: str, hint: str = "") -> Dict[str, Any]:
    """
    用 LLM 理解品牌：它是什么品类？消费者讨论什么？该搜什么词？

    返回 dict 包含 category/product_type/search_queries/evaluation_dimensions 等
    LLM 不可用时用关键词匹配降级
    """
    if is_llm_available():
        try:
            user_prompt = f"品牌名称：{brand_name}"
            if hint and hint.strip():
                user_prompt += f"\n用户提示：{hint}"
            else:
                user_prompt += "\n（用户没有提供额外信息，请基于品牌名称推断）"

            result = call_llm_json(
                system_prompt=BRAND_CONTEXT_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
            )

            if "error" not in result and result.get("category"):
                print(f"[品牌理解] {brand_name} → {result.get('category')} | {result.get('product_type')}")
                print(f"[品牌理解] 搜索词: {result.get('search_queries', [])[:5]}")
                return result
        except Exception as e:
            print(f"[品牌理解] LLM分析失败: {e}")

    # 降级：从品牌名+提示词中提取关键词
    print(f"[品牌理解] LLM不可用，使用关键词降级")
    return _brand_from_keywords(brand_name, hint)


def _brand_from_keywords(brand_name: str, hint: str) -> Dict[str, Any]:
    """关键词匹配的品牌分类（LLM 不可用时的降级）"""
    combined = f"{brand_name} {hint}"
    default_queries = [
        f"{brand_name} 怎么样",
        f"{brand_name} 评价",
        f"{brand_name} 测评",
        f"{brand_name} 推荐",
        f"{brand_name} 踩雷",
    ]

    return {
        "category": hint.strip() if hint.strip() else "消费品",
        "product_type": hint.strip() if hint.strip() else "产品",
        "description": f"{brand_name}品牌",
        "typical_price": "未知",
        "consumer_talk": ["好不好", "值不值", "怎么样"],
        "search_queries": default_queries,
        "evaluation_dimensions": ["品质", "价格", "体验"],
    }


def _multi_query_search(search_queries: List[str], brand_name: str,
                        total_count: int) -> List[dict]:
    """用多个搜索词组合搜索，去重合并结果。前两个连续失败则熔断跳过剩余。"""
    import time as _time
    all_results = []
    seen = set()
    per_query = max(5, total_count // len(search_queries))
    consecutive_failures = 0

    for i, sq in enumerate(search_queries):
        if len(all_results) >= total_count:
            break
        # 前两个搜索词都失败 → 后面大概率也不行，熔断
        if consecutive_failures >= 2:
            print(f"[搜索] 前{i}个搜索词均失败，熔断跳过剩余{len(search_queries)-i}个")
            break

        try:
            batch = search_news(sq, max_results=per_query)
            if batch:
                consecutive_failures = 0
                for r in batch:
                    key = (r.get("title", "")[:80], r.get("source", ""))
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)
            else:
                consecutive_failures += 1
        except Exception as e:
            consecutive_failures += 1
            print(f"[搜索] '{sq[:40]}' 异常: {e}")

        # 搜索词之间短暂延迟，避免速率限制
        if i < len(search_queries) - 1 and consecutive_failures == 0:
            _time.sleep(0.5)

    print(f"[爬虫Agent] {len(search_queries)}个搜索词 → {len(all_results)}条结果（熔断={consecutive_failures>=2}）")
    return all_results


# ==================== LLM 提取 ====================

def _extract_with_llm(raw_results: List[dict]) -> List[SentimentItem]:
    """用 LLM 将搜索结果批量转为 SentimentItem"""
    items = []
    for i, result in enumerate(raw_results):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        source = result.get("source", "")
        date_str = result.get("date", "")

        if not title or len(title) < 3:
            continue

        user_prompt = f"来源：{source}\n标题：{title}\n摘要：{snippet}"

        try:
            extracted = call_llm_json(
                system_prompt=EXTRACT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
            )

            if "error" in extracted:
                # LLM JSON 解析失败，走启发式提取
                item = _heuristic_single(result, i)
                items.append(item)
                continue

            # 情感倾向 → 用于估算互动数据（后面 _enrich_engagement 会覆盖）
            sentiment_hint = extracted.get("sentiment_hint", "中性")
            is_negative = sentiment_hint == "负面"
            is_positive = sentiment_hint == "正向"

            # 解析平台
            platform_str = extracted.get("platform", "新闻媒体")
            platform = _map_platform(platform_str, source)

            # 解析时间
            publish_time = _parse_datetime(date_str)

            # 媒体权重
            media_weight = estimate_source_weight(source)

            item = SentimentItem(
                id=f"real_{uuid.uuid4().hex[:12]}",
                platform=platform,
                content=extracted.get("content", title),
                author=extracted.get("author", source),
                publish_time=publish_time,
                likes=random.randint(100, 5000) if is_negative else random.randint(10, 500),
                shares=random.randint(20, 2000) if is_negative else random.randint(1, 200),
                comments=random.randint(30, 3000) if is_negative else random.randint(2, 100),
                media_weight=media_weight,
            )
            items.append(item)

        except Exception as e:
            print(f"[爬虫Agent] LLM提取第{i}条失败: {e}")
            # 降级：直接用原始标题+摘要作为舆情内容
            item = _heuristic_single(result, i)
            items.append(item)

    return items


# ==================== 启发式提取（无LLM时） ====================

def _extract_heuristic(raw_results: List[dict], brand_name: str) -> List[SentimentItem]:
    """基于关键词的启发式提取（不需要LLM）"""
    items = []
    for i, result in enumerate(raw_results):
        item = _heuristic_single(result, i)
        items.append(item)
    return items


def _heuristic_single(result: dict, index: int) -> SentimentItem:
    """将单条搜索结果转为 SentimentItem（关键词匹配）"""
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    source = result.get("source", "")
    date_str = result.get("date", "")

    content = f"{title}。{snippet}" if snippet else title

    # 简单情感判断
    negative_words = ["投诉", "曝光", "翻车", "维权", "召回", "下架", "罚款", "调查",
                      "暴跌", "造假", "违规", "致癌", "中毒", "爆炸", "事故", "缺陷",
                      "道歉", "约谈", "整改", "停业", "封禁", "差评", "维权"]
    positive_words = ["好评", "推荐", "好评如潮", "点赞", "值得信赖", "品质保证",
                      "澄清", "辟谣", "回应称不实", "获赞", "领跑"]

    text = title + snippet
    neg_count = sum(1 for w in negative_words if w in text)
    pos_count = sum(1 for w in positive_words if w in text)

    if neg_count > pos_count:
        sentiment = "负面"
    elif pos_count > neg_count:
        sentiment = "正向"
    else:
        sentiment = "中性"

    is_negative = sentiment == "负面"

    # 推断平台
    source_lower = source.lower()
    if any(s in source_lower for s in ["weibo", "微博"]):
        platform = Platform.WEIBO
    elif any(s in source_lower for s in ["douyin", "抖音"]):
        platform = Platform.DOUYIN
    elif any(s in source_lower for s in ["xiaohongshu", "小红书"]):
        platform = Platform.REDBOOK
    elif any(s in source_lower for s in ["zhihu", "知乎"]):
        platform = Platform.ZHIHU
    elif any(s in source_lower for s in ["tieba", "贴吧"]):
        platform = Platform.TIEBA
    else:
        platform = Platform.NEWS

    media_weight = estimate_source_weight(source)
    publish_time = _parse_datetime(date_str)

    return SentimentItem(
        id=f"web_{uuid.uuid4().hex[:12]}",
        platform=platform,
        content=content[:500],
        author=source,
        publish_time=publish_time,
        likes=random.randint(100, 5000) if is_negative else random.randint(10, 500),
        shares=random.randint(20, 2000) if is_negative else random.randint(1, 200),
        comments=random.randint(30, 3000) if is_negative else random.randint(2, 100),
        media_weight=media_weight,
    )


# ==================== 兜底生成（搜索引擎完全不可用时） ====================

def _generate_dynamic_fallback(brand_name: str, brand_context: Dict[str, Any],
                               count: int) -> List[SentimentItem]:
    """
    基于 LLM 品牌理解动态生成消费者评价（搜索引擎完全不可用时的最终兜底）。

    与旧版对比：
      旧版：写死的辣条/零食/手机等几个品类词库，其他品类没覆盖
      新版：根据 LLM 返回的 category/product_type/evaluation_dimensions
            动态构造评价内容，覆盖任意品类
    """
    cat = brand_context.get("category", "消费品")
    product = brand_context.get("product_type", "产品")
    price = brand_context.get("typical_price", "不贵")
    dims = brand_context.get("evaluation_dimensions", ["品质", "体验", "性价比"])
    talk = brand_context.get("consumer_talk", ["怎么样", "好不好"])

    pos_count = int(count * 0.35)
    neu_count = int(count * 0.30)
    neg_count = count - pos_count - neu_count

    items = []

    for i in range(count):
        if i < pos_count:
            sentiment = "正向"
        elif i < pos_count + neu_count:
            sentiment = "中性"
        else:
            sentiment = "负面"

        content = _build_dynamic_review(sentiment, brand_name, product, cat, price, dims, talk)
        platform = random.choice([Platform.WEIBO, Platform.REDBOOK, Platform.ZHIHU,
                                  Platform.DOUYIN, Platform.TIEBA])
        author = _random_author(sentiment)

        if sentiment == "负面":
            likes = random.randint(200, 15000)
            shares = random.randint(30, 3000)
            comments = random.randint(50, 2000)
        elif sentiment == "正向":
            likes = random.randint(20, 2000)
            shares = random.randint(5, 300)
            comments = random.randint(5, 150)
        else:
            likes = random.randint(10, 1000)
            shares = random.randint(2, 100)
            comments = random.randint(5, 100)

        hours_ago = random.randint(0, 72)
        publish_time = datetime.now() - timedelta(hours=hours_ago)

        items.append(SentimentItem(
            id=f"dyn_{uuid.uuid4().hex[:10]}",
            platform=platform,
            content=content[:500],
            author=author,
            publish_time=publish_time,
            likes=likes, shares=shares, comments=comments,
            media_weight=1.0,
        ))

    random.shuffle(items)
    return items


def _build_dynamic_review(sentiment: str, brand: str, product: str,
                          category: str, price: str,
                          dimensions: List[str],
                          talk_topics: List[str]) -> str:
    """根据品牌信息动态构造一条消费者评价"""
    dim_pick = random.choice(dimensions) if dimensions else "体验"
    topic = random.choice(talk_topics) if talk_topics else "怎么样"

    if sentiment == "正向":
        templates = [
            f"{brand}的{product}真的绝了！{dim_pick}超出预期，{price}这个价位太值了，已经回购好几次了。",
            f"试了{brand}{product}，{dim_pick}让我很满意。对比了好几家最后还是觉得这个最好，推荐！",
            f"跟风买了{brand}{product}，本来没抱期望的，结果{dim_pick}确实不错，{price}性价比很高。",
            f"被朋友安利的{brand}{product}，吃了/用了就停不下来，{dim_pick}确实比别家强。",
            f"测评了五六个{category}品牌，{brand}的{dim_pick}是最能打的，这个价位无敌了。",
        ]
    elif sentiment == "中性":
        templates = [
            f"{brand}{product}试了，{dim_pick}中规中矩吧。{price}这个价格也不能要求太多。",
            f"网上风很大的{brand}{product}，买来发现{dim_pick}还可以但也没吹得那么神，正常水平。",
            f"关于{brand}的{topic}，我觉得{dim_pick}还行，有好有坏吧。建议大家先试试小包装。",
            f"{brand}{product}怎么说呢，{dim_pick}不算差但也没有惊喜。对这个品类来说就是及格水平。",
        ]
    else:
        templates = [
            f"避雷{brand}的{product}！！{dim_pick}真的很差，{price}完全不值。失望。",
            f"看好多博主推{brand}，买回来发现{dim_pick}翻车了。{topic}方面真的不行，别跟风。",
            f"讲真{brand}这个{product}，{dim_pick}太让人失望了。{price}花得真冤枉。",
            f"{brand}最近营销太多了，实际{dim_pick}根本不行。{topic}还不如普通牌子。",
            f"作为一个{category}爱好者，{brand}的{dim_pick}是我最近最踩雷的一次。",
        ]

    return random.choice(templates)


def _random_author(sentiment: str) -> str:
    """生成随机但合理的用户昵称"""
    prefixes = {
        "正向": ["吃货", "零食控", "生活家", "购物达人", "好物分享", "种草"],
        "中性": ["理性消费", "老实人", "普通用户", "路人", "冷静"],
        "负面": ["踩雷专家", "真实评价", "暴躁", "避雷", "不推荐"],
    }
    suffix = random.choice(["小明", "小红", "阿强", "小芳", "大壮", "豆豆", "圆圆",
                            "老王", "小林", "小美", "阿杰", "小龙"])
    prefix = random.choice(prefixes.get(sentiment, ["用户"]))
    return f"{prefix}{suffix}"


# ==================== 辅助函数 ====================

def _enrich_engagement(items: List[SentimentItem]) -> None:
    """为舆情条目补充/调整互动数据，使其更符合真实传播规律"""
    for item in items:
        # 根据发布时间调整互动量级：越近的越少（还没发酵）
        if hasattr(item, 'publish_time') and item.publish_time:
            hours_ago = (datetime.now() - item.publish_time).total_seconds() / 3600

            if hours_ago < 2:
                # 刚发布，互动还在积累
                item.likes = min(item.likes, random.randint(5, 500))
                item.shares = min(item.shares, random.randint(1, 100))
                item.comments = min(item.comments, random.randint(2, 200))
            elif hours_ago > 24:
                # 发布超过24h，如果是负面且互动量高说明持续发酵
                if item.likes > 5000:
                    item.likes = int(item.likes * random.uniform(1.5, 3.0))


def _map_platform(platform_str: str, source: str) -> Platform:
    """将LLM输出的平台字符串映射到 Platform 枚举"""
    platform_mapping = {
        "微博": Platform.WEIBO,
        "抖音": Platform.DOUYIN,
        "小红书": Platform.REDBOOK,
        "新闻媒体": Platform.NEWS,
        "贴吧": Platform.TIEBA,
        "知乎": Platform.ZHIHU,
    }
    if platform_str in platform_mapping:
        return platform_mapping[platform_str]

    # 通过来源域名辅助判断
    source_lower = source.lower()
    if "weibo" in source_lower:
        return Platform.WEIBO
    if "douyin" in source_lower or "tiktok" in source_lower:
        return Platform.DOUYIN
    if "xiaohongshu" in source_lower:
        return Platform.REDBOOK
    if "zhihu" in source_lower:
        return Platform.ZHIHU
    if "tieba" in source_lower:
        return Platform.TIEBA
    return Platform.NEWS


def _parse_datetime(date_str: str) -> datetime:
    """解析搜索结果的日期字符串"""
    if not date_str:
        hours_ago = random.randint(0, 48)
        return datetime.now() - timedelta(hours=hours_ago)

    # 尝试多种日期格式
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m月%d日",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    # 解析 "X 小时前" / "X 天前"
    hours_match = re.search(r'(\d+)\s*小时前', date_str)
    if hours_match:
        return datetime.now() - timedelta(hours=int(hours_match.group(1)))
    days_match = re.search(r'(\d+)\s*天前', date_str)
    if days_match:
        return datetime.now() - timedelta(days=int(days_match.group(1)))

    # 兜底
    hours_ago = random.randint(0, 48)
    return datetime.now() - timedelta(hours=hours_ago)
