# ============================================================
# backend/search.py - 搜索引擎模块
# 支持 Google Custom Search API → Playwright 浏览器搜索（降级）
# 设计理念：API 优先，不可用时自动切换真实浏览器搜索
# ============================================================

import os
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# ==================== 搜索配置 ====================

SEARCH_CONFIG = {
    # Google Custom Search API（推荐，速度快）
    "google_api_key": os.environ.get("GOOGLE_API_KEY", ""),
    "google_cx": os.environ.get("GOOGLE_CX", ""),  # Search Engine ID
    # SerpAPI（备选）
    "serpapi_key": os.environ.get("SERPAPI_KEY", ""),
    # HTTP 代理（国内访问 SerpAPI 可能需要）
    # 格式: "http://127.0.0.1:7890" 或 "socks5://127.0.0.1:1080"
    "proxy": os.environ.get("HTTP_PROXY", os.environ.get("HTTPS_PROXY", "")),
    # 默认参数
    "max_results": 20,
    "language": "zh-CN",
}


# ==================== 搜索结果数据模型 ====================

class SearchResult:
    """统一搜索结果格式"""
    def __init__(self, title: str, snippet: str, source: str, url: str,
                 date: Optional[str] = None, source_type: str = "web"):
        self.title = title
        self.snippet = snippet
        self.source = source
        self.url = url
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.source_type = source_type

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
            "url": self.url,
            "date": self.date,
        }


# ==================== 方案1：Google Custom Search API ====================

def _google_api_search(query: str, max_results: int = 20) -> List[SearchResult]:
    """
    使用 Google Custom Search JSON API。
    免费额度：每天 100 次查询。
    需要设置 GOOGLE_API_KEY 和 GOOGLE_CX 环境变量。
    """
    import requests

    api_key = SEARCH_CONFIG["google_api_key"]
    cx = SEARCH_CONFIG["google_cx"]

    if not api_key or not cx:
        raise ValueError("未配置 GOOGLE_API_KEY 和 GOOGLE_CX")

    results = []
    # Google CSE 每次最多返回 10 条，需要翻页
    for start in range(1, min(max_results + 1, 31), 10):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "lr": "lang_zh-CN",
            "num": min(10, max_results - len(results)),
            "start": start,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if "items" not in data:
            break

        for item in data["items"]:
            results.append(SearchResult(
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                source=_extract_domain(item.get("link", "")),
                url=item.get("link", ""),
                date=_parse_google_date(item.get("pagemap", {})),
            ))

        if len(results) >= max_results:
            break

    return results


# ==================== 方案2：SerpAPI（Google 搜索的付费代理） ====================

def _retry_request(url: str, params: dict, max_retries: int = 2,
                   timeout: int = 15) -> Optional[Any]:
    """
    带指数退避重试的 HTTP 请求。
    返回 JSON 数据，失败返回 None。
    """
    import requests
    last_error = None

    proxies = None
    proxy_url = SEARCH_CONFIG.get("proxy", "")
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, proxies=proxies)
            # 先检查 HTTP 状态码
            if resp.status_code == 401 or resp.status_code == 403:
                print(f"[搜索] API Key无效或额度用尽 (HTTP {resp.status_code})")
                print(f"[搜索] {resp.text[:200]}")
                return None
            if resp.status_code == 429:
                wait = min((attempt + 1) * 10, 30)
                print(f"[搜索] 速率限制(429)，{wait}s 后重试...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"[搜索] HTTP {resp.status_code}: {resp.text[:150]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            last_error = e
            err_str = str(e)[:100]
            if "10054" in err_str or "reset" in err_str.lower():
                if not proxy_url:
                    print("[搜索] 连接被重置 - 国内访问 SerpAPI 可能被墙")
                    print("[搜索] 解决: 在 .env 设置 HTTP_PROXY=http://127.0.0.1:代理端口")
                else:
                    print(f"[搜索] 代理 {proxy_url} 连接失败，请检查代理")
            if attempt < max_retries - 1:
                wait = min((attempt + 1) * 2, 8)
                print(f"[搜索] {wait}s 后重试({attempt+1}/{max_retries})...")
                time.sleep(wait)
        except requests.exceptions.Timeout:
            last_error = "timeout"
            if attempt < max_retries - 1:
                wait = min((attempt + 1) * 2, 8)
                print(f"[搜索] 超时，{wait}s 后重试({attempt+1}/{max_retries})...")
                time.sleep(wait)
        except Exception as e:
            last_error = str(e)[:100]
            if attempt < max_retries - 1:
                print(f"[搜索] 异常({last_error[:60]})，{2}s 后重试...")
                time.sleep(2)

    return None


def _serpapi_search(query: str, max_results: int = 20) -> List[SearchResult]:
    """
    使用 SerpAPI 搜索，多策略覆盖真实用户讨论。
    策略1: 普通网页搜索（含论坛/评测/社交媒体）
    策略2: 新闻搜索（品牌重大事件）
    免费额度：每月 100 次。
    """
    api_key = SEARCH_CONFIG["serpapi_key"]
    if not api_key:
        raise ValueError("未配置 SERPAPI_KEY")

    all_results = []
    seen_urls = set()

    # === 策略1：普通网页搜索（覆盖论坛、评测、小红书、知乎等） ===
    data = _retry_request(
        "https://serpapi.com/search",
        params={"q": query, "hl": "zh-CN", "gl": "cn",
                "api_key": api_key, "num": min(max_results, 20)},
    )
    if data:
        for r in data.get("organic_results", [])[:max_results]:
            url_str = r.get("link", "")
            if url_str in seen_urls:
                continue
            seen_urls.add(url_str)
            all_results.append(SearchResult(
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                source=_extract_domain(url_str),
                url=url_str,
                date=r.get("date", ""),
                source_type="web",
            ))

    # === 策略2：新闻搜索（补充品牌重大事件） ===
    if len(all_results) < max_results:
        time.sleep(1)
        data = _retry_request(
            "https://serpapi.com/search",
            params={"q": query, "tbm": "nws", "hl": "zh-CN", "gl": "cn",
                    "api_key": api_key, "num": min(max_results - len(all_results), 20)},
        )
        if data:
            for r in data.get("news_results", [])[:max_results - len(all_results)]:
                url_str = r.get("link", "")
                if url_str in seen_urls:
                    continue
                seen_urls.add(url_str)
                all_results.append(SearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    source=r.get("source", ""),
                    url=url_str,
                    date=r.get("date", ""),
                    source_type="news",
                ))

    return all_results[:max_results]


# ==================== 方案3：Playwright 浏览器搜索（免费、无需 API Key） ====================

def _playwright_search(query: str, max_results: int = 20,
                       search_engine: str = "google") -> List[SearchResult]:
    """
    用真实浏览器引擎打开搜索引擎并提取结果。
    完全免费，无需 API Key，但速度较慢（3-8秒/次）。

    search_engine: "google" | "bing" | "baidu"
    需要安装: pip install playwright && playwright install chromium
    """
    from playwright.sync_api import sync_playwright
    # 延迟导入，不影响没有安装 playwright 的用户

    results = []

    if search_engine == "google":
        search_url = f"https://www.google.com/search?q={query}&tbm=nws&lr=lang_zh-CN&num={max_results}"
        title_selector = "div.n0jPhd"
        snippet_selector = "div.GI74Re"
        source_selector = "span.NUnG9d span"
        wait_selector = "div.SoaBEf, div.MjjYud"
    elif search_engine == "bing":
        search_url = f"https://www.bing.com/news/search?q={query}&cc=cn"
        title_selector = "a.title"
        snippet_selector = "div.snippet"
        source_selector = "div.source"
        wait_selector = "div.news-card"
    else:  # baidu (百度新闻搜索)
        search_url = f"https://www.baidu.com/s?wd={query}&tn=news"
        title_selector = "h3.c-title a"
        snippet_selector = "div.c-summary"
        source_selector = "div.c-author"
        wait_selector = "div.result"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=15000)

            # 等待搜索结果加载
            try:
                page.wait_for_selector(wait_selector, timeout=8000)
            except Exception:
                pass  # 可能被拦截或没有结果

            # 滚动加载更多
            for _ in range(min(max_results // 5, 4)):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)

            # 提取标题
            titles = page.query_selector_all(title_selector)
            snippets = page.query_selector_all(snippet_selector)
            sources = page.query_selector_all(source_selector) or [None] * len(titles)

            for i in range(min(len(titles), max_results)):
                try:
                    title = titles[i].inner_text().strip()
                except Exception:
                    title = ""

                try:
                    snippet = snippets[i].inner_text().strip() if i < len(snippets) else ""
                except Exception:
                    snippet = ""

                try:
                    source = sources[i].inner_text().strip() if i < len(sources) and sources[i] else _extract_domain(
                        page.url)
                except Exception:
                    source = ""

                if title:
                    results.append(SearchResult(
                        title=title[:200],
                        snippet=snippet[:500],
                        source=source[:100],
                        url="",
                        source_type="news",
                    ))

        except Exception as e:
            print(f"[搜索] Playwright 搜索异常: {e}")
        finally:
            browser.close()

    return results


# ==================== 统一搜索入口（自动选择策略） ====================

def search_news(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    统一搜索入口。

    自动多词搜索策略：对消费品/品牌类查询，同时搜索多个角度
      - 基础搜索："{brand} {keyword}"
      - 消费者视角："{brand} 好吃吗" / "{brand} 测评" / "{brand} 踩雷"
      - 讨论视角："{brand} 怎么样" / "{brand} 推荐"

    结果去重合并，覆盖新闻+论坛+评测+社交讨论。
    """
    all_results = []
    seen = set()

    # 从原始 query 中拆分 brand 和 keyword
    parts = query.split(maxsplit=1)
    brand = parts[0] if parts else query

    # 构建多个搜索词，覆盖不同信息维度
    search_queries = [
        query,                          # 原始查询
        f"{query} 测评",                 # 评测
        f"{query} 好吃吗 怎么样",         # 消费者讨论
        f"{query} 踩雷 吐槽 推荐",        # 正负面评价
        f"{brand} 怎么样 评价",           # 品牌评价
    ]

    def _do_search(q: str, n: int) -> List[SearchResult]:
        """内部：按优先级尝试各种搜索引擎"""
        results = []

        # Priority 1: SerpAPI
        if SEARCH_CONFIG["serpapi_key"]:
            try:
                results = _serpapi_search(q, n)
                if results:
                    print(f"[搜索] SerpAPI '{q[:40]}' → {len(results)} 条")
                    return results
            except Exception as e:
                print(f"[搜索] SerpAPI 失败: {e}")

        # Priority 2: Google CSE
        if not results and SEARCH_CONFIG["google_api_key"] and SEARCH_CONFIG["google_cx"]:
            try:
                results = _google_api_search(q, n)
                if results:
                    print(f"[搜索] Google CSE '{q[:40]}' → {len(results)} 条")
                    return results
            except Exception as e:
                print(f"[搜索] Google CSE 失败: {e}")

        # Priority 3: Playwright
        if not results:
            try:
                for engine in ["google", "bing", "baidu"]:
                    try:
                        results = _playwright_search(q, n, engine)
                        if results:
                            break
                    except Exception:
                        continue
            except ImportError:
                pass
            except Exception as e:
                print(f"[搜索] Playwright 失败: {e}")

        return results

    # 逐词搜索，每个词分配一些额度，直到凑够
    per_query = max(8, max_results // len(search_queries))
    for sq in search_queries:
        if len(all_results) >= max_results:
            break
        batch = _do_search(sq, per_query)
        for r in batch:
            key = r.title[:80]  # 按标题去重
            if key not in seen:
                seen.add(key)
                all_results.append(r)

    print(f"[搜索] 多词搜索完成：{len(all_results)} 条去重结果")
    return [r.to_dict() for r in all_results[:max_results]]


# ==================== 辅助函数 ====================

def _extract_domain(url: str) -> str:
    """从 URL 中提取域名"""
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1) if match else url[:50]


def _parse_google_date(pagemap: dict) -> str:
    """从 Google 搜索结果中提取日期"""
    try:
        metatags = pagemap.get("metatags", [{}])
        for meta in metatags:
            for key in ["article:published_time", "datePublished", "pubdate"]:
                if key in meta:
                    return meta[key][:10]
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")


def estimate_source_weight(source: str) -> float:
    """
    根据来源域名估算媒体权威权重（用于风险计算）。
    知名媒体权重 > 2.0，普通网站 1.0-1.5，自媒体 < 1.0
    """
    HIGH_AUTHORITY = {
        "xinhuanet.com": 2.8, "people.com.cn": 2.8, "cctv.com": 3.0,
        "sina.com.cn": 2.0, "qq.com": 2.0, "163.com": 1.8,
        "sohu.com": 1.8, "thepaper.cn": 2.5, "caixin.com": 2.5,
        "bbc.com": 2.5, "reuters.com": 2.8, "bloomberg.com": 2.8,
        "ft.com": 2.5, "wsj.com": 2.8, "nytimes.com": 2.5,
    }
    source_lower = source.lower()
    for domain, weight in HIGH_AUTHORITY.items():
        if domain in source_lower:
            return weight
    return 1.2  # 默认一般来源


# ==================== 诊断工具 ====================

def diagnose() -> Dict[str, Any]:
    """Diagnose search configuration. Run: python -c 'from backend.search import diagnose; print(diagnose())'"""
    import requests

    result = {
        "serpapi_key_configured": bool(SEARCH_CONFIG["serpapi_key"]),
        "serpapi_key_prefix": SEARCH_CONFIG["serpapi_key"][:8] + "..." if SEARCH_CONFIG["serpapi_key"] else "not set",
        "google_api_configured": bool(SEARCH_CONFIG["google_api_key"] and SEARCH_CONFIG["google_cx"]),
        "proxy_configured": bool(SEARCH_CONFIG["proxy"]),
        "proxy": SEARCH_CONFIG["proxy"] if SEARCH_CONFIG["proxy"] else "not set",
    }

    if not SEARCH_CONFIG["serpapi_key"]:
        result["diagnosis"] = "SERPAPI_KEY not configured"
        return result

    proxies = None
    if SEARCH_CONFIG["proxy"]:
        proxies = {"http": SEARCH_CONFIG["proxy"], "https": SEARCH_CONFIG["proxy"]}

    # Test 1: basic connectivity
    try:
        resp = requests.get("https://serpapi.com", timeout=10, proxies=proxies)
        result["connectivity"] = "OK (HTTP %d)" % resp.status_code
    except Exception as e:
        result["connectivity"] = "FAIL: %s" % str(e)[:100]

    # Test 2: API call (minimal query)
    try:
        resp = requests.get("https://serpapi.com/search", params={
            "q": "test", "api_key": SEARCH_CONFIG["serpapi_key"], "num": 1,
        }, timeout=15, proxies=proxies)
        if resp.status_code == 200:
            data = resp.json()
            n = len(data.get("organic_results", []))
            result["api_test"] = "OK - %d results" % n
        elif resp.status_code == 401:
            result["api_test"] = "FAIL: API Key invalid (401)"
            result["diagnosis"] = "SERPAPI_KEY is invalid or expired"
        elif resp.status_code == 403:
            result["api_test"] = "FAIL: Permission denied (403)"
            result["diagnosis"] = "SERPAPI_KEY has no search permission or quota exhausted (free: 100/month)"
        else:
            result["api_test"] = "HTTP %d: %s" % (resp.status_code, resp.text[:200])
    except requests.exceptions.ConnectionError as e:
        result["api_test"] = "FAIL: Connection error - %s" % str(e)[:100]
        if "10054" in str(e) or "reset" in str(e):
            result["diagnosis"] = "Network blocked (GFW). Set HTTP_PROXY in .env or switch to Google CSE"
    except Exception as e:
        result["api_test"] = "Exception: %s" % str(e)[:100]

    return result
