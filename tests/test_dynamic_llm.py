"""dynamic_llm 测试：模型名前缀规范化与实例缓存。"""

from __future__ import annotations

from prompt2vid import dynamic_llm


def test_model_without_prefix_gets_openai_prefix(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_LLM_BASE_URL", "http://gw:4001/v1")
    monkeypatch.delenv("PROMPT2VID_MODEL", raising=False)
    assert dynamic_llm._normalize_model("qwen3.8-27b", "http://gw:4001/v1") == "openai/qwen3.8-27b"
    # 已带前缀原样保留
    assert dynamic_llm._normalize_model("openai/gpt-4o-mini", "http://gw:4001/v1") == "openai/gpt-4o-mini"
    assert dynamic_llm._normalize_model("deepseek/deepseek-chat", "http://gw:4001/v1") == "deepseek/deepseek-chat"


def test_model_without_prefix_no_base_url_unchanged():
    assert dynamic_llm._normalize_model("qwen3.8-27b", "") == "qwen3.8-27b"


def test_build_llm_normalizes_and_caches(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_MODEL", "qwen3.8-27b")
    monkeypatch.setenv("PROMPT2VID_LLM_BASE_URL", "http://gw:4001/v1")
    monkeypatch.setenv("PROMPT2VID_LLM_API_KEY", "sk-test")
    dynamic_llm._LLM_CACHE.clear()

    llm1 = dynamic_llm.build_llm("script_writer")
    llm2 = dynamic_llm.build_llm("image_generator")
    # 相同配置复用同一实例（9 个 agent 不重复建 LLM 客户端）
    assert llm1 is llm2
    # 前缀补齐后进入缓存键（crewai LLM 的 .model 属性会剥掉 openai/ 前缀，
    # 但构造时收到的是补齐后的名字，LiteLLM 才能正确路由）
    assert ("openai/qwen3.8-27b", "http://gw:4001/v1", "sk-test", 0.7) in dynamic_llm._LLM_CACHE

    # 不同配置不复用
    monkeypatch.setenv("PROMPT2VID_MODEL__SCRIPT_WRITER", "openai/gpt-4o")
    llm3 = dynamic_llm.build_llm("script_writer")
    assert llm3 is not llm1


def test_resolve_model_agent_override(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("PROMPT2VID_MODEL__SCRIPT_WRITER", "openai/gpt-4o")
    assert dynamic_llm.resolve_model("script_writer") == "openai/gpt-4o"
    assert dynamic_llm.resolve_model("image_generator") == "openai/gpt-4o-mini"
