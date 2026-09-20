# ============================================================
# backend/llm_client.py - 大模型统一调用封装
# 支持 OpenAI / DeepSeek / 通义千问 / 本地模型等所有 OpenAI 兼容 API
# 特性：多模型切换、指数退避重试、优雅降级
# ============================================================

import os
import json
import time
import re
from typing import Optional, Dict, Any

# ==================== 配置 ====================

# ==================== 默认使用 DeepSeek ====================
# 设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
# 也可设置 LLM_BASE_URL / LLM_MODEL 切换其他模型

LLM_CONFIG = {
    "api_key": os.environ.get("DEEPSEEK_API_KEY")
           or os.environ.get("OPENAI_API_KEY", ""),
    "base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "default_model": os.environ.get("LLM_MODEL", "deepseek-chat"),
    "fast_model": os.environ.get("LLM_FAST_MODEL", "deepseek-chat"),
    "max_retries": 3,
    "timeout": 60,
}

# 其他模型切换（取消注释即可）：
# LLM_CONFIG["base_url"] = "https://api.openai.com/v1"
# LLM_CONFIG["default_model"] = "gpt-4o"
# LLM_CONFIG["fast_model"] = "gpt-4o-mini"
#
# LLM_CONFIG["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 通义千问
# LLM_CONFIG["default_model"] = "qwen-plus"


# ==================== 客户端初始化（延迟加载） ====================

_client = None
_available = None  # None=未检测, True=可用, False=不可用


def _get_client():
    """获取 OpenAI 客户端实例（延迟初始化）"""
    global _client, _available
    if _available is None:
        try:
            from openai import OpenAI
            _client = OpenAI(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
                timeout=LLM_CONFIG["timeout"],
            )
            # 快速测试连接
            _client.models.list()
            _available = True
            print(f"[LLM] 已连接到 {LLM_CONFIG['base_url']}，默认模型：{LLM_CONFIG['default_model']}")
        except Exception as e:
            _available = False
            print(f"[LLM] 连接失败（{str(e)[:80]}），将使用降级模式（规则引擎）")
            print("[LLM] 请设置环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL")
    return _client if _available else None


def is_llm_available() -> bool:
    """检测 LLM 是否可用"""
    return _get_client() is not None


# ==================== 核心调用函数 ====================

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    response_format: Optional[str] = None,  # "json_object" 强制 JSON 输出
) -> str:
    """
    LLM 统一调用入口，带指数退避重试。

    参数：
        system_prompt: 系统提示词（定义角色和规则）
        user_prompt:   用户提示词（具体任务输入）
        model:         模型名，默认用 LLM_CONFIG["default_model"]
        temperature:   温度 0.0~2.0，分析类任务建议 0.1~0.3
        max_tokens:    最大输出 token 数
        response_format: 设为 "json_object" 强制返回 JSON

    返回：
        LLM 响应的文本内容

    异常：
        连接失败时抛出 RuntimeError，调用方应捕获并降级
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("LLM 服务不可用，请检查 API Key 和网络连接")

    model = model or LLM_CONFIG["default_model"]

    kwargs = dict(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(LLM_CONFIG["max_retries"]):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            return content if content else ""
        except Exception as e:
            err_msg = str(e)
            # 速率限制 → 等久一点
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                wait = min((attempt + 1) * 10, 60)
                print(f"[LLM] 速率限制，{wait}s 后重试...")
                time.sleep(wait)
            elif attempt < LLM_CONFIG["max_retries"] - 1:
                wait = min(2 ** attempt, 8)
                print(f"[LLM] 调用失败（{err_msg[:60]}），{wait}s 后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM 调用失败（已重试 {LLM_CONFIG['max_retries']} 次）: {err_msg}")

    raise RuntimeError("LLM 调用失败：未知错误")


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    调用 LLM 并强制返回 JSON 对象。

    带自动修复：如果 LLM 返回的不是纯 JSON（被 markdown 包裹等），
    尝试用正则提取 ```json...``` 代码块。

    返回：
        dict: 解析后的 JSON 对象，失败时返回 {"error": "..."}
    """
    try:
        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or LLM_CONFIG["fast_model"],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
        )
        return _safe_parse_json(raw)
    except Exception as e:
        print(f"[LLM] JSON 调用失败: {e}，尝试非 JSON 模式...")
        # 降级：不用 response_format，手动提取 JSON
        try:
            raw = call_llm(
                system_prompt=system_prompt + "\n请只返回JSON，不要包含任何其他文字。",
                user_prompt=user_prompt,
                model=model or LLM_CONFIG["default_model"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _safe_parse_json(raw)
        except Exception as e2:
            print(f"[LLM] 降级调用也失败了: {e2}")
            return {"error": str(e2)}


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    """安全解析 JSON，处理常见的格式问题"""
    if not raw:
        return {"error": "空响应"}

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 { 到最后一个 }
    brace_start = raw.find('{')
    brace_end = raw.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(raw[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return {"error": "JSON 解析失败", "raw": raw[:500]}


# ==================== 便捷函数 ====================

def call_llm_batch(
    system_prompt: str,
    user_prompts: list[str],
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> list[str]:
    """
    批量调用（顺序执行，非并行）。
    用于批量情感分析等场景。
    """
    results = []
    for i, prompt in enumerate(user_prompts):
        try:
            result = call_llm(system_prompt, prompt, model=model, temperature=temperature)
            results.append(result)
        except Exception as e:
            print(f"[LLM] 批量调用 [{i}/{len(user_prompts)}] 失败: {e}")
            results.append("")  # 失败返回空字符串
    return results


def check_connection() -> Dict[str, Any]:
    """健康检查：测试 LLM 连接状态"""
    if not is_llm_available():
        return {
            "status": "unavailable",
            "message": "LLM 服务不可用，请设置 OPENAI_API_KEY 环境变量",
        }
    try:
        start = time.time()
        call_llm(
            system_prompt="你是一个助手。",
            user_prompt="回复'OK'",
            model=LLM_CONFIG["fast_model"],
            max_tokens=10,
        )
        elapsed = round((time.time() - start) * 1000)
        return {
            "status": "healthy",
            "model": LLM_CONFIG["default_model"],
            "base_url": LLM_CONFIG["base_url"],
            "latency_ms": elapsed,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "message": str(e)[:200],
        }
