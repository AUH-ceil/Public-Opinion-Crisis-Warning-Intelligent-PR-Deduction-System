# ============================================================
# backend/qdrant_client.py - Qdrant向量库客户端（极简RAG实现）
# 向量化使用 sklearn TfidfVectorizer（纯本地，无需下载模型）
# 支持 Qdrant 未启动时自动降级为本地内存检索
# ============================================================

import os
import json
from typing import List, Dict, Any, Optional
import uuid
import numpy as np

# ==================== 配置（从 .env 读取） ====================
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = "crisis_cases"
VECTOR_SIZE = 256       # TF-IDF 特征维度

# ==================== 向量化器（sklearn Tfidf，无需下载） ====================

_tfidf = None           # TfidfVectorizer 单例
_known_texts = []       # 已拟合的文本列表（增量更新用）
_fitted = False         # 是否已拟合


def _get_tfidf():
    """延迟初始化 TfidfVectorizer（中文用字符级+二元组）"""
    global _tfidf
    if _tfidf is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # 字符级 + 1-2字符组合，适配中文
        _tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 2),
            max_features=VECTOR_SIZE,
        )
    return _tfidf


def _fit_tfidf_if_needed(texts: List[str]):
    """如果未拟合，用已有文本初始化 TF-IDF 词汇表"""
    global _fitted, _known_texts
    if not _fitted and texts:
        _known_texts = list(texts)
        _get_tfidf().fit(_known_texts)
        _fitted = True


def text_to_vector(text: str) -> List[float]:
    """
    文本 → 向量（TF-IDF 字符级 ngram）
    纯本地计算，不依赖任何外部模型下载
    """
    if not _fitted:
        # 未拟合时用空文本先拟合一个初始词汇表
        _fit_tfidf_if_needed([text])
    vec = _get_tfidf().transform([text]).toarray()[0]
    # 补齐到 VECTOR_SIZE（确保维度固定）
    if len(vec) < VECTOR_SIZE:
        padded = np.zeros(VECTOR_SIZE)
        padded[:len(vec)] = vec
        return padded.tolist()
    return vec.tolist()


# ==================== Qdrant 客户端（可选，离线也能用） ====================

_qdrant = None
_qdrant_available = None  # None=未检测, True=可用, False=不可用


def _get_qdrant():
    """延迟连接 Qdrant，失败则返回 None"""
    global _qdrant, _qdrant_available
    if _qdrant_available is None:
        try:
            from qdrant_client import QdrantClient as QC
            _qdrant = QC(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
            _qdrant.get_collections()  # 测试连接
            _qdrant_available = True
            print(f"[Qdrant] 已连接到 {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            _qdrant_available = False
            print(f"[Qdrant] 未连接（{e}），使用本地内存检索模式")
    return _qdrant if _qdrant_available else None


# ==================== 本地内存存储（Qdrant离线时降级使用） ====================

_LOCAL_STORE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "vector_store.json")

_local_store: Dict[str, dict] = {}  # case_id → {vector, payload}
_loaded_from_disk = False


def _persist_local_store():
    """将本地存储持久化到磁盘 JSON 文件"""
    try:
        data = {}
        for cid, entry in _local_store.items():
            data[cid] = {
                "vector": entry["vector"],
                "payload": entry["payload"],
                "uuid": entry.get("uuid", ""),
            }
        os.makedirs(os.path.dirname(_LOCAL_STORE_FILE), exist_ok=True)
        with open(_LOCAL_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"[本地存储] 持久化失败: {e}")


def _load_local_store():
    """从磁盘 JSON 文件恢复本地存储"""
    global _local_store, _loaded_from_disk
    if _loaded_from_disk:
        return
    _loaded_from_disk = True

    try:
        if os.path.exists(_LOCAL_STORE_FILE):
            with open(_LOCAL_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cid, entry in data.items():
                # 确保向量是 list[float]
                vec = entry.get("vector", [])
                if isinstance(vec, list) and len(vec) > 0:
                    _local_store[cid] = {
                        "vector": vec,
                        "payload": entry.get("payload", {}),
                        "uuid": entry.get("uuid", ""),
                    }
            print(f"[本地存储] 从磁盘恢复 {len(_local_store)} 条数据")
    except Exception as e:
        print(f"[本地存储] 加载失败: {e}")


def _local_search(query_vector: List[float], limit: int) -> List[dict]:
    """本地余弦相似度检索（Qdrant 不可用时的降级方案）"""
    _load_local_store()
    import math
    qv = np.array(query_vector)
    results = []
    for cid, entry in _local_store.items():
        ev = np.array(entry["vector"])
        # 余弦相似度
        dot = np.dot(qv, ev)
        norm = np.linalg.norm(qv) * np.linalg.norm(ev)
        score = dot / norm if norm > 0 else 0.0
        results.append({"id": cid, "score": float(score), "payload": entry["payload"]})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


# ==================== 公开 API（与之前兼容） ====================

def init_qdrant():
    """初始化 Qdrant 集合（如果可用），否则用本地存储"""
    client = _get_qdrant()
    if client is None:
        print(f"[Qdrant] 离线模式，使用本地内存存储")
        _load_local_store()
        return

    from qdrant_client.models import Distance, VectorParams
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if COLLECTION_NAME in names:
        print(f"[Qdrant] 集合 '{COLLECTION_NAME}' 已存在，跳过创建")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[Qdrant] 集合 '{COLLECTION_NAME}' 创建成功")


def _str_to_uuid(s: str) -> str:
    """将字符串转为确定性UUID，Qdrant要求UUID格式"""
    import hashlib
    h = hashlib.md5(s.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def upsert_case(case_id: str, text: str, payload: Dict[str, Any]):
    """插入/更新一条案例"""
    vector = text_to_vector(text)
    client = _get_qdrant()
    uid = _str_to_uuid(case_id)

    if client is not None:
        from qdrant_client.models import PointStruct
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=uid, vector=vector, payload=payload)]
        )
    else:
        _local_store[case_id] = {"vector": vector, "payload": payload, "uuid": uid}
        _persist_local_store()


def search_similar_cases(query_text: str, limit: int = 5,
                          category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    相似案例检索
    query_text: 当前舆情文本 → 转为向量 → 检索最相似的 N 条
    """
    query_vector = text_to_vector(query_text)
    client = _get_qdrant()

    if client is not None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qf = None
        if category_filter:
            qf = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category_filter))])

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            query_filter=qf,
        ).points
        return [
            {"case_id": str(r.id), "score": round(r.score, 4), **r.payload}
            for r in results
        ]
    else:
        results = _local_search(query_vector, limit)
        # 按分类过滤
        if category_filter:
            results = [r for r in results if r["payload"].get("category") == category_filter]
        return [
            {"case_id": r["id"], "score": round(r["score"], 4), **r["payload"]}
            for r in results
        ]


def save_analysis(brand_name: str, category: str, risk_level: str,
                  risk_index: float, negative_ratio: float,
                  sentiment_keywords: list, pr_actions: list,
                  similar_case_titles: list) -> str:
    """
    将一次分析结果存入向量库，供后续 RAG 检索。

    存入的内容包括：品牌、品类、风险等级、核心问题、
    公关策略摘要——下次搜到类似问题时就能参考。
    """
    import uuid as _uuid
    from datetime import datetime as _dt

    case_id = f"analysis_{_dt.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"

    # 构建向量化文本（用于相似度匹配）
    keywords_str = "、".join(sentiment_keywords[:8]) if sentiment_keywords else "暂无"
    text = (
        f"品牌：{brand_name}。品类：{category}。"
        f"风险等级：{risk_level}，风险指数：{risk_index}。"
        f"核心负面关键词：{keywords_str}。"
    )

    # 构建 payload（用于展示详情）
    title = f"{brand_name}舆情分析（{category}）" if category else f"{brand_name}舆情分析"
    description = (
        f"本次分析发现{brand_name}面临{risk_level}风险（指数{risk_index}），"
        f"负面舆情占比{negative_ratio*100:.0f}%。"
        f"消费者主要反馈集中在：{keywords_str}。"
    ) if sentiment_keywords else (
        f"本次分析发现{brand_name}面临{risk_level}风险（指数{risk_index}），"
        f"负面舆情占比{negative_ratio*100:.0f}%。"
    )

    # 公关策略摘要作为 resolution
    steps_text = "；".join(pr_actions[:6]) if pr_actions else "未生成具体策略"
    resolution = f"【{risk_level}风险应对】{steps_text}"

    # 引用相似案例
    if similar_case_titles:
        resolution += f"。参考历史案例：{'、'.join(similar_case_titles[:3])}"

    payload = {
        "title": title,
        "category": category or "综合",
        "description": description,
        "resolution": resolution,
        "risk_level": risk_level,
        "brand_name": brand_name,
        "risk_index": risk_index,
        "saved_at": _dt.now().isoformat(),
        "source": "system_analysis",  # 标记为系统自动生成
    }

    upsert_case(case_id, text, payload)

    where = "Qdrant" if _get_qdrant() is not None else "本地"
    print(f"[向量库] 分析结果已存入{where}：{title}（{risk_level}风险）")

    return case_id


def sync_mysql_to_qdrant():
    """
    手动将 MySQL history_records 全量同步到 Qdrant。
    用于 MySQL 新增数据后刷新向量库索引。
    """
    samples = _load_samples_from_mysql()
    if not samples:
        print("[Qdrant] MySQL 无数据可同步")
        return 0

    all_texts = [f"{c['title']}。{c['description']}" for c in samples]
    _fit_tfidf_if_needed(all_texts)

    count = 0
    for case in samples:
        text = f"{case['title']}。{case['description']}"
        upsert_case(case["case_id"], text, {
            "title": case["title"],
            "category": case["category"],
            "description": case["description"],
            "resolution": case["resolution"],
            "risk_level": case["risk_level"],
        })
        count += 1

    where = "Qdrant" if _get_qdrant() is not None else "本地内存"
    print(f"[{where}] MySQL 同步完成: {count} 条记录")
    return count


def list_all_cases() -> List[Dict[str, Any]]:
    """列出向量库中所有存储的案例数据"""
    client = _get_qdrant()
    results = []

    if client is not None:
        try:
            count = client.count(COLLECTION_NAME).count
            points, _ = client.scroll(COLLECTION_NAME, limit=max(count, 50))
            for p in points:
                pld = p.payload or {}
                pld["_id"] = str(p.id)
                results.append(pld)
        except Exception as e:
            print(f"[Qdrant] 查询失败: {e}")
    else:
        for cid, entry in _local_store.items():
            pld = dict(entry.get("payload", {}))
            pld["_id"] = cid
            results.append(pld)

    return results


def print_all_cases():
    """在控制台打印向量库中所有数据（方便调试查看）"""
    cases = list_all_cases()
    print(f"\n{'='*60}")
    print(f"  向量库数据一览（共 {len(cases)} 条）")
    print(f"{'='*60}")
    for i, c in enumerate(cases):
        print(f"\n[{i+1}] {c.get('title', '?')}")
        print(f"    ID: {c.get('_id', '?')}")
        print(f"    分类: {c.get('category', '?')} | 风险: {c.get('risk_level', '?')}")
        src = c.get('source', c.get('brand_name', ''))
        if src:
            print(f"    来源: {src}")
        res = c.get('resolution', '')
        if res:
            print(f"    方案: {res[:120]}")
    print(f"\n{'='*60}\n")
    return cases


def delete_collection():
    """删除集合（调试用）"""
    client = _get_qdrant()
    if client is not None:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    _local_store.clear()
    global _fitted, _known_texts, _tfidf
    _fitted = False
    _known_texts = []
    _tfidf = None


def seed_qdrant_samples():
    """
    初始化向量库数据。
    优先从 MySQL history_records 表同步（单一数据源），
    MySQL 不可用时用内置样例数据兜底。
    """
    client = _get_qdrant()
    if client is not None:
        try:
            count = client.count(COLLECTION_NAME).count
            if count > 0:
                print(f"[Qdrant] 向量库已有 {count} 条数据，跳过填充")
                return
        except Exception:
            init_qdrant()
    else:
        if _local_store:
            print(f"[本地存储] 已有 {len(_local_store)} 条数据，跳过填充")
            return

    # ===== 优先从 MySQL 同步 =====
    samples = _load_samples_from_mysql()
    if samples:
        print(f"[Qdrant] 从 MySQL history_records 同步 {len(samples)} 条数据")
    else:
        # MySQL 没数据，用内置样例兜底
        samples = _get_builtin_samples()
        print(f"[Qdrant] MySQL 无数据，使用内置样例 {len(samples)} 条")

    # 收集文本拟合 TF-IDF 并逐条入库
    all_texts = [f"{c['title']}。{c['description']}" for c in samples]
    _fit_tfidf_if_needed(all_texts)

    for case in samples:
        text = f"{case['title']}。{case['description']}"
        upsert_case(case["case_id"], text, {
            "title": case["title"],
            "category": case["category"],
            "description": case["description"],
            "resolution": case["resolution"],
            "risk_level": case["risk_level"],
        })

    where = "Qdrant" if _get_qdrant() is not None else "本地内存"
    print(f"[{where}] 数据填充完成: {len(samples)} 条历史案例")


def _load_samples_from_mysql() -> list:
    """从 MySQL history_records 表加载数据，转为 Qdrant 格式"""
    try:
        from backend.database import get_all_history_records
        rows = get_all_history_records()
        if not rows:
            return []

        samples = []
        for i, row in enumerate(rows):
            risk_map = {4: "特级", 3: "高", 2: "中", 1: "低"}
            risk_idx = row.get("risk_index", 0)
            if risk_idx >= 75:
                risk_level = "特级"
            elif risk_idx >= 50:
                risk_level = "高"
            elif risk_idx >= 25:
                risk_level = "中"
            else:
                risk_level = "低"

            samples.append({
                "case_id": f"mysql_seed_{row.get('id', i)}",
                "title": str(row.get("event_name", "")),
                "category": str(row.get("category", "产品质量")),
                "description": (
                    f"{row.get('event_name', '')}。"
                    f"首发平台：{row.get('platform', '未知')}，"
                    f"风险指数：{row.get('risk_index', 0)}，"
                    f"负面分值：{row.get('negative_score', 0)}，"
                    f"热搜耗时：{row.get('hot_search_hours', 0)}小时。"
                ),
                "resolution": str(row.get("resolution", "")),
                "risk_level": risk_level,
            })
        return samples
    except Exception as e:
        print(f"[Qdrant] 从 MySQL 加载数据失败: {e}")
        return []


def _get_builtin_samples() -> list:
    """内置样例数据（MySQL 不可用时的兜底）"""
    return [
        {"case_id": "case_001", "title": "某连锁奶茶店食品安全事件", "category": "食品安全",
         "description": "消费者在奶茶中发现异物，照片上传微博后引发大量转发。品牌初期回应不当，导致舆情24小时内登上热搜，门店营业额下降70%。最终通过CEO公开道歉、邀请第三方检测、全面整改卫生流程、赔偿消费者等措施平息。",
         "resolution": "第一步：2小时内下架问题产品并发布初步声明；第二步：24小时内CEO录制道歉视频；第三步：邀请第三方卫生检测机构入驻；第四步：公布整改时间表并接受消费者监督；第五步：补偿方案：全额退款+1000元赔偿+全年免单券。",
         "risk_level": "特级"},
        {"case_id": "case_002", "title": "某手机品牌电池自燃事件", "category": "产品质量",
         "description": "多位用户反映手机充电时电池过热自燃，短视频累计播放超2亿次。品牌初期否认，后因央视报道被迫承认，全球召回50万台设备。",
         "resolution": "第一步：立即成立专项调查组；第二步：24小时内发布全球召回公告；第三步：联合第三方检测机构公布调查结果；第四步：CEO公开道歉并承诺升级品控标准；第五步：全额退款+换新机。",
         "risk_level": "特级"},
        {"case_id": "case_003", "title": "某五星酒店服务纠纷事件", "category": "服务纠纷",
         "description": "住客曝光酒店前台态度恶劣、房间卫生不达标，小红书笔记获10万+点赞。酒店回应迟缓且傲慢，被旅游局约谈。",
         "resolution": "第一步：1小时内人工道歉；第二步：免去涉事员工职务并公示；第三步：邀请住客代表参观整改后的酒店；第四步：全面服务培训；第五步：全额退款+免费住宿3晚。",
         "risk_level": "高"},
        {"case_id": "case_004", "title": "某电商平台大规模售假事件", "category": "产品质量",
         "description": "消费者协会报告称某平台第三方卖家售假率达30%，微博话题阅读量超5亿，多个品牌终止合作，市值单日蒸发50亿元。",
         "resolution": "第一步：24小时内封禁涉事店铺并公布名单；第二步：设立10亿元假货举报基金；第三步：引入区块链溯源系统；第四步：假一赔十升级为假一赔百；第五步：每月发布打假报告。",
         "risk_level": "特级"},
        {"case_id": "case_005", "title": "某快餐品牌后厨卫生事件", "category": "食品安全",
         "description": "暗访记者曝光快餐品牌后厨蟑螂遍地、食材过期，视频播放量超5亿。股价单日下跌15%，多地监管部门介入。",
         "resolution": "第一步：2小时内承认问题并道歉；第二步：涉事门店停业整顿；第三步：全国门店安装明厨亮灶监控；第四步：邀请消费者参观后厨；第五步：建立食品安全举报奖励制度。",
         "risk_level": "特级"},
        {"case_id": "case_006", "title": "某化妆品品牌过敏事故", "category": "产品质量",
         "description": "多位消费者反映面霜过敏，小红书相关笔记超2万篇。品牌初期声称符合国标，后被检测出违规添加成分。",
         "resolution": "第一步：下架问题批次产品；第二步：公布第三方成分检测报告；第三步：完善过敏提示标签；第四步：设立消费者皮肤健康基金；第五步：赔偿医疗费用+全额退款。",
         "risk_level": "高"},
        {"case_id": "case_007", "title": "某航空公司超售拒载事件", "category": "服务纠纷",
         "description": "航空公司超售强制拒绝已登机旅客，安保暴力拖拽，视频疯传全球，引发公关灾难。",
         "resolution": "第一步：CEO在4小时内发布公开道歉声明；第二步：调整超售政策承诺永不强拖乘客；第三步：提高补偿标准至1万元；第四步：全员服务培训；第五步：设立旅客权益保障基金。",
         "risk_level": "特级"},
        {"case_id": "case_008", "title": "日用品牌广告争议事件", "category": "服务纠纷",
         "description": "日用品广告被指歧视女性，微博话题阅读量超3亿，多家女性权益组织发起抵制。",
         "resolution": "第一步：2小时内撤下争议广告；第二步：发布诚恳道歉声明；第三步：邀请女性员工参与广告审核流程；第四步：向女性公益组织捐赠100万元；第五步：建立广告内容多元化审核委员会。",
         "risk_level": "中"},
    ]
