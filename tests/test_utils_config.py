"""
utils/config.py 测试
====================
Config 单例的加载、合并、查询行为测试。
"""

import pytest
import threading
from unittest.mock import patch

from src.utils.config import Config


# Config 单例在每个测试后被 reset_config_singleton autouse fixture 重置


class TestConfigSingleton:
    def test_singleton_behavior(self):
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_thread_safety(self):
        """两个线程应获得同一个实例。"""
        instances = []

        def get_instance():
            instances.append(Config())

        t1 = threading.Thread(target=get_instance)
        t2 = threading.Thread(target=get_instance)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert instances[0] is instances[1]


class TestConfigGet:
    def test_top_level_key(self):
        c = Config()
        result = c.get("llm")
        assert isinstance(result, dict) or result is None

    def test_dot_path_lookup(self):
        c = Config()
        # 直接设置内部状态进行测试
        c._config = {"llm": {"model": "gpt-4", "api_key": "sk-test"}}
        assert c.get("llm.model") == "gpt-4"
        assert c.get("llm.api_key") == "sk-test"

    def test_nested_missing_key_default(self):
        c = Config()
        c._config = {"llm": {"model": "gpt-4"}}
        assert c.get("llm.nonexistent", "default") == "default"

    def test_missing_top_key_default(self):
        c = Config()
        c._config = {}
        assert c.get("nonexistent.key", 42) == 42

    def test_intermediate_non_dict_returns_default(self):
        c = Config()
        c._config = {"llm": "not_a_dict"}
        assert c.get("llm.model", "fallback") == "fallback"

    def test_empty_key_returns_default(self):
        c = Config()
        c._config = {"a": 1, "b": 2}
        result = c.get("", "default_full")
        assert result == "default_full"


class TestDeepMerge:
    def test_shallow_merge(self):
        c = Config()
        result = c._deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        c = Config()
        base = {"llm": {"model": "gpt-4", "temperature": 0.7}}
        override = {"llm": {"model": "gpt-5"}}
        result = c._deep_merge(base, override)
        assert result["llm"]["model"] == "gpt-5"
        assert result["llm"]["temperature"] == 0.7  # 保留未覆盖的嵌套键

    def test_override_empty_base(self):
        c = Config()
        result = c._deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_override_adds_new_nested(self):
        c = Config()
        base = {"section_a": {"x": 1}}
        override = {"section_b": {"y": 2}}
        result = c._deep_merge(base, override)
        assert result["section_a"] == {"x": 1}
        assert result["section_b"] == {"y": 2}


class TestLoadFromEnv:
    def test_api_key_mapped_to_llm(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        c = Config()
        result = c._load_from_env()
        assert result["llm"]["api_key"] == "sk-test-key"

    def test_api_base_mapped_to_llm(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_BASE", "https://custom.api.com/v1")
        c = Config()
        result = c._load_from_env()
        assert result["llm"]["api_base"] == "https://custom.api.com/v1"

    def test_default_model_mapped(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_MODEL", "deepseek-v4")
        c = Config()
        result = c._load_from_env()
        assert result["llm"]["model"] == "deepseek-v4"

    def test_embedding_model_mapped(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        c = Config()
        result = c._load_from_env()
        assert result["embedding"]["model"] == "text-embedding-3-large"

    def test_memory_dirs_mapped(self, monkeypatch):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma")
        monkeypatch.setenv("MEMORY_FILE_DIR", "/tmp/memory")
        c = Config()
        result = c._load_from_env()
        assert result["memory"]["long_term"]["persist_dir"] == "/tmp/chroma"
        assert result["memory"]["file_store"]["base_dir"] == "/tmp/memory"

    def test_log_level_mapped(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        c = Config()
        result = c._load_from_env()
        assert result["system"]["log_level"] == "DEBUG"

    def test_missing_env_no_error(self):
        """环境变量缺失不应导致错误。"""
        c = Config()
        with patch.dict("os.environ", {}, clear=True):
            result = c._load_from_env()
            assert isinstance(result, dict)


class TestConfigGetAll:
    def test_returns_copy(self):
        c = Config()
        c._config = {"key": "value"}
        result = c.get_all()
        result["key"] = "modified"
        assert c._config["key"] == "value"  # 原始不受影响


class TestConfigReload:
    def test_reload_resets_config(self):
        c = Config()
        original = c.get_all()
        c._config = {"custom": "data"}
        c.reload()
        # reload 后不应包含我们手动设置的数据
        assert c.get("custom") is None
