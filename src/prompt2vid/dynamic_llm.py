"""按 agent 动态选择 LLM。

支持通过环境变量为不同 agent 覆盖模型：
- PROMPT2VID_MODEL：全局默认模型（LiteLLM 格式，如 `openai/gpt-4o-mini`）
- PROMPT2VID_MODEL__<AGENT_NAME>：按 agent 覆盖（agent 名大写、点换双下划线）
  例如 PROMPT2VID_MODEL__SCRIPT_WRITER=openai/gpt-4o

同时支持 PROMPT2VID_LLM_BASE_URL / PROMPT2VID_LLM_API_KEY / PROMPT2VID_LLM_TEMPERATURE。

模型名规范：LiteLLM 要求 `提供商/模型名` 格式；配置了 OpenAI 兼容网关
（PROMPT2VID_LLM_BASE_URL）而模型名不含 `/` 时自动补 `openai/` 前缀，
避免第一次调用才报路由错误。相同配置的 agent 复用同一个 LLM 实例。
"""

from __future__ import annotations

import os

from crewai import LLM

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7

_LLM_CACHE: dict[tuple, LLM] = {}


def _agent_env_suffix(agent_name: str) -> str:
    """把 agent 名转换为环境变量后缀：大写、点换双下划线。"""
    return agent_name.upper().replace(".", "__")


def resolve_model(agent_name: str | None = None) -> str:
    """解析某个 agent 应使用的模型名。"""
    if agent_name:
        override = os.getenv(
            f"PROMPT2VID_MODEL__{_agent_env_suffix(agent_name)}", ""
        ).strip()
        if override:
            return override
    return os.getenv("PROMPT2VID_MODEL", "").strip() or DEFAULT_MODEL


def _normalize_model(model: str, base_url: str) -> str:
    """模型名补 LiteLLM 提供商前缀。

    已带前缀（含 `/`）原样保留；配了 OpenAI 兼容网关而无前缀时补
    `openai/`（如 `qwen3.8-27b` → `openai/qwen3.8-27b`）。
    """
    if "/" in model:
        return model
    if base_url:
        return f"openai/{model}"
    return model


def build_llm(agent_name: str | None = None) -> LLM:
    """构建某个 agent 的 LLM 实例（CrewAI LiteLLM 封装，按配置缓存复用）。"""
    base_url = os.getenv("PROMPT2VID_LLM_BASE_URL", "").strip()
    model = _normalize_model(resolve_model(agent_name), base_url)

    kwargs: dict = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url

    api_key = os.getenv("PROMPT2VID_LLM_API_KEY", "").strip()
    if api_key:
        kwargs["api_key"] = api_key

    try:
        temperature = float(os.getenv("PROMPT2VID_LLM_TEMPERATURE", "").strip())
    except ValueError:
        temperature = DEFAULT_TEMPERATURE
    kwargs["temperature"] = temperature

    cache_key = (model, base_url, api_key, temperature)
    if cache_key not in _LLM_CACHE:
        _LLM_CACHE[cache_key] = LLM(**kwargs)
    return _LLM_CACHE[cache_key]
