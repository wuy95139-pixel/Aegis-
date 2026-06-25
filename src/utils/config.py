"""
配置加载器
=========
从 config.yaml 和环境变量加载配置，合并为统一配置字典。

设计决策：
  - config.yaml 存储默认值和非敏感配置
  - .env 存储敏感信息 (API Key 等)
  - 运行时通过 merge_configs() 合并，env 优先级高于 yaml
"""

import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv

# 自动加载 .env 文件
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
else:
    # 尝试从当前工作目录加载
    load_dotenv()


class Config:
    """
    全局配置单例
    使用方法: Config().get("llm.model")
    """

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    _lock = threading.Lock()

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: Optional[str] = None):
        """加载并合并配置"""
        # 1. 加载 YAML 配置
        if config_path is None:
            # 项目根目录: src/utils/config.py → parent.parent.parent = project root
            config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"

        yaml_config = {}
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}

        # 2. 从环境变量提取配置
        env_config = self._load_from_env()

        # 3. 合并：环境变量优先级更高
        self._config = self._deep_merge(yaml_config, env_config)

    def _load_from_env(self) -> Dict[str, Any]:
        """从环境变量构建配置字典"""
        env_cfg: Dict[str, Any] = {}

        # LLM
        if os.getenv("OPENAI_API_KEY"):
            env_cfg.setdefault("llm", {})
            env_cfg["llm"]["api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_API_BASE"):
            env_cfg.setdefault("llm", {})
            env_cfg["llm"]["api_base"] = os.getenv("OPENAI_API_BASE")
        if os.getenv("DEFAULT_MODEL"):
            env_cfg.setdefault("llm", {})
            env_cfg["llm"]["model"] = os.getenv("DEFAULT_MODEL")

        # Embedding
        if os.getenv("EMBEDDING_MODEL"):
            env_cfg.setdefault("embedding", {})
            env_cfg["embedding"]["model"] = os.getenv("EMBEDDING_MODEL")

        # Memory
        if os.getenv("CHROMA_PERSIST_DIR"):
            env_cfg.setdefault("memory", {}).setdefault("long_term", {})
            env_cfg["memory"]["long_term"]["persist_dir"] = os.getenv("CHROMA_PERSIST_DIR")
        if os.getenv("MEMORY_FILE_DIR"):
            env_cfg.setdefault("memory", {}).setdefault("file_store", {})
            env_cfg["memory"]["file_store"]["base_dir"] = os.getenv("MEMORY_FILE_DIR")

        # System
        if os.getenv("LOG_LEVEL"):
            env_cfg.setdefault("system", {})
            env_cfg["system"]["log_level"] = os.getenv("LOG_LEVEL")

        return env_cfg

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并两个字典，override 优先级更高"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        通过点号路径获取配置值
        例如: Config().get("llm.model") 返回 "deepseek-v4-pro"
        """
        keys = key_path.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_all(self) -> Dict[str, Any]:
        """返回完整配置字典"""
        return self._config.copy()

    def get_data_path(self, subpath: str) -> Path:
        """
        解析数据路径，相对于配置的 data_dir。
        如果 subpath 是绝对路径则直接返回。

        Args:
            subpath: 相对于 data_dir 的路径，如 "chroma_db", "memory", "tasks.json"

        Returns:
            解析后的绝对路径
        """
        sub_path = Path(subpath)
        if sub_path.is_absolute():
            return sub_path
        data_dir = self.get("system.data_dir", "./data")
        return (Path(data_dir) / sub_path).resolve()

    @staticmethod
    def get_project_root() -> Path:
        """返回项目根目录的绝对路径"""
        return _PROJECT_ROOT

    def reload(self, config_path: Optional[str] = None):
        """重新加载配置"""
        self._config = {}
        self._load(config_path)
