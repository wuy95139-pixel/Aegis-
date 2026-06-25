"""
长期记忆模块
============
基于 ChromaDB (可选 FAISS) 的向量存储，持久化关键信息。

设计决策：
  - 每次对话结束后，由 MemoryAgent 总结关键信息并存入
  - 存储的是 "记忆卡片" (MemoryEntry)，包含内容、标签、重要性评分
  - 检索时使用混合检索：语义相似度 + 标签过滤 + 重要性加权

可扩展点：
  - 切换到 FAISS: 修改 backend 参数，FAISS 适合超大规模 (>100万) 向量
  - 支持多模态嵌入: 图片也可以生成 embedding 存储
  - 记忆衰减: 根据时间衰减重要性评分
"""

import os
import time
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.models.schemas import MemoryEntry

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    长期记忆 — ChromaDB 向量数据库

    使用示例:
        ltm = LongTermMemory(persist_dir="./data/chroma_db")
        ltm.store(MemoryEntry(id="m1", content="...", source="chat_001"))
        results = ltm.search("用户偏好", top_k=5)
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "aegis_long_term_memory",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Args:
            persist_dir: ChromaDB 持久化目录
            collection_name: 集合名称
            embedding_model: 嵌入模型名 (OpenAI 格式)
            api_key: OpenAI API Key (用于在线嵌入)
        """
        # 转换为绝对路径 (Windows 兼容)
        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._is_persistent = True

        # 确保持久化目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        # 初始化 ChromaDB 客户端 — Windows 兼容 + 重试
        self.client = self._create_persistent_client()

        # 嵌入函数 — 使用 OpenAI 兼容 API
        # 回退策略: 默认嵌入模型 (all-MiniLM-L6-v2) 与 OpenAI embedding 不兼容，
        # 混用会导致向量空间不一致，语义搜索悄悄失效。
        # 允许回退基于环境变量: AEGIS_ALLOW_DEFAULT_EMBEDDING=true
        self._using_default_embedding = False
        try:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
                api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                model_name=embedding_model,
            )
        except Exception as e:
            allow_default = os.getenv("AEGIS_ALLOW_DEFAULT_EMBEDDING", "").lower() == "true"
            if allow_default:
                logger.error(
                    f"OpenAIEmbedding init failed: {e}. "
                    "Falling back to DefaultEmbeddingFunction (all-MiniLM-L6-v2). "
                    "WARNING: Embedding vectors will be incompatible with previously stored vectors! "
                    "Semantic search may return garbage results. Set OPENAI_API_KEY to fix."
                )
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
                self._using_default_embedding = True
            else:
                logger.error(
                    f"OpenAIEmbedding init failed: {e}. "
                    "Embedding is REQUIRED for semantic search but unavailable. "
                    "Long-term semantic search will be disabled. "
                    "Set OPENAI_API_KEY to fix, or AEGIS_ALLOW_DEFAULT_EMBEDDING=true for degraded fallback."
                )
                self.embedding_fn = None

        # 获取或创建集合（embedding_fn 为 None 时跳过，语义搜索不可用）
        if self.embedding_fn is not None:
            try:
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.error(f"ChromaDB collection init failed: {e}")
                fallback_name = f"{collection_name}_fallback"
                try:
                    self.collection = self.client.get_or_create_collection(
                        name=fallback_name,
                        embedding_function=self.embedding_fn,
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception as e2:
                    logger.error(f"ChromaDB fallback collection also failed: {e2}")
                    self.collection = None
        else:
            self.collection = None
            logger.warning("ChromaDB collection not created: no embedding function available")

        if not self._is_persistent:
            logger.warning(
                "⚠️  ChromaDB 运行在非持久化模式！"
                "重启后向量数据将丢失，但文件存储 (FileStore) 不受影响。"
                "长期记忆可通过 FileStore 检索恢复。"
            )

        coll_info = f"{self.collection.name} ({self.collection.count()} entries)" if self.collection else "disabled"
        logger.info(
            f"LongTermMemory initialized: persist_dir={self.persist_dir}, "
            f"persistent={self._is_persistent}, "
            f"collection={coll_info}"
        )

    def _create_persistent_client(self):
        """
        创建 ChromaDB PersistentClient，带 Windows 兼容和重试逻辑。

        在 Windows 上，SQLite 文件可能因以下原因失败：
          - 文件被其他进程锁定（上次未正常退出）
          - 路径包含特殊字符
          - 防病毒软件拦截

        策略：尝试 PersistentClient → 清理锁文件后重试 → EphemeralClient
        """
        for attempt in range(3):
            try:
                client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
                # 验证客户端可用
                client.heartbeat()
                logger.debug(f"ChromaDB PersistentClient OK: {self.persist_dir}")
                return client
            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"ChromaDB PersistentClient attempt {attempt + 1}/3 failed: {error_msg[:120]}"
                )
                if attempt < 2:
                    time.sleep(0.5)
                    # Windows: 尝试清理 SQLite 锁文件
                    self._cleanup_sqlite_locks()

        # 3 次重试都失败，使用 EphemeralClient 但记录错误
        self._is_persistent = False
        logger.error(
            "❌ ChromaDB PersistentClient 初始化失败（已重试3次）！\n"
            "   向量检索将不可用，但文件存储 (FileStore) 仍可正常工作。\n"
            "   建议：关闭其他使用同一 ChromaDB 目录的进程后重启。\n"
            f"   ChromaDB 目录: {self.persist_dir}"
        )
        return chromadb.EphemeralClient(
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _cleanup_sqlite_locks(self):
        """清理 ChromaDB SQLite 残留锁文件（Windows 兼容）"""
        try:
            db_dir = Path(self.persist_dir)
            for lock_file in db_dir.glob("**/*.lock"):
                try:
                    lock_file.unlink()
                    logger.debug(f"Removed stale lock file: {lock_file}")
                except OSError:
                    pass
        except Exception:
            logger.debug("SQLite lock cleanup skipped", exc_info=True)

    def store(self, entry: MemoryEntry) -> str:
        """
        存储一条记忆

        Args:
            entry: 记忆条目

        Returns:
            记忆 ID
        """
        if not self.collection:
            logger.debug("LongTermMemory store skipped: no collection available")
            return entry.id or str(uuid.uuid4())

        entry_id = entry.id or str(uuid.uuid4())

        # 构建元数据 (ChromaDB metadata 只支持 str/int/float/bool)
        metadata = {
            "source": entry.source,
            "tags": ",".join(entry.tags),
            "importance": entry.importance,
            "created_at": entry.created_at.isoformat(),
        }

        # 带重试的存储 (ChromaDB 在 Windows 上可能因 SQLite 锁/集合不一致而间歇失败)
        for attempt in range(3):
            try:
                if entry.embedding:
                    self.collection.add(
                        ids=[entry_id],
                        documents=[entry.content],
                        embeddings=[entry.embedding],
                        metadatas=[metadata],
                    )
                else:
                    self.collection.add(
                        ids=[entry_id],
                        documents=[entry.content],
                        metadatas=[metadata],
                    )
                logger.debug(f"Stored memory: id={entry_id}, source={entry.source}")
                return entry_id
            except Exception as e:
                error_msg = str(e)
                if attempt < 2:
                    logger.warning(
                        f"ChromaDB add failed (attempt {attempt + 1}/3): {error_msg[:120]}. Retrying..."
                    )
                    time.sleep(0.3)
                    # 尝试重建集合引用 (可能是 stale collection handle)
                    try:
                        self.collection = self.client.get_or_create_collection(
                            name=self.collection_name,
                            embedding_function=self.embedding_fn,
                            metadata={"hnsw:space": "cosine"},
                        )
                    except Exception as _e:
                        logger.debug("ChromaDB collection refresh failed during retry: %s", _e)
                else:
                    logger.error(f"ChromaDB add failed after 3 attempts: {error_msg[:200]}")
                    raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        搜索相关记忆

        Args:
            query: 查询文本
            top_k: 返回条数
            tags: 按标签过滤 (AND 逻辑)
            source: 按来源过滤

        Returns:
            排序后的记忆条目列表
        """
        if not self.collection:
            logger.debug("LongTermMemory search skipped: no collection available")
            return []

        # 构建过滤条件
        where_filter: Optional[Dict] = None
        conditions = []

        if tags:
            # ChromaDB 字符串包含过滤
            for tag in tags:
                conditions.append({"tags": {"$contains": tag}})

        if source:
            conditions.append({"source": {"$eq": source}})

        if conditions:
            where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        # 带重试的搜索 (ChromaDB 在 Windows 上可能因 SQLite 锁/集合过期而间歇失败)
        for attempt in range(2):
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, 20),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
                break
            except Exception as e:
                if attempt < 1:
                    logger.warning(f"ChromaDB query failed (attempt {attempt + 1}/2): {e}. Retrying...")
                    time.sleep(0.3)
                    try:
                        self.collection = self.client.get_or_create_collection(
                            name=self.collection_name,
                            embedding_function=self.embedding_fn,
                            metadata={"hnsw:space": "cosine"},
                        )
                    except Exception as _e:
                        logger.debug("ChromaDB collection refresh failed during query retry: %s", _e)
                else:
                    logger.error(f"ChromaDB query failed after 2 attempts: {e}")
                    return []

        # 解析结果
        entries = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                entries.append(MemoryEntry(
                    id=mem_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    source=metadata.get("source", "unknown"),
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    importance=float(metadata.get("importance", 0.5)),
                    similarity_score=max(0.0, 1.0 - distance),  # 余弦距离转相似度，clamp 负值
                ))

        logger.debug(f"Search query='{query[:50]}...' returned {len(entries)} results")
        return entries

    def delete(self, entry_id: str) -> None:
        """删除一条记忆"""
        if not self.collection:
            logger.debug("LongTermMemory delete skipped: no collection available")
            return
        self.collection.delete(ids=[entry_id])
        logger.debug(f"Deleted memory: id={entry_id}")

    def forget_source(self, source: str) -> None:
        """删除某个来源的所有记忆 (如某个对话的所有记忆)"""
        if not self.collection:
            logger.debug("LongTermMemory forget_source skipped: no collection available")
            return
        self.collection.delete(where={"source": {"$eq": source}})
        logger.info(f"Forgot all memories from source={source}")

    def count(self) -> int:
        """返回记忆总数"""
        return self.collection.count() if self.collection else 0

    def list_by_source(self, source: str, limit: int = 50) -> List[MemoryEntry]:
        """
        列出某个来源的所有记忆 (不经过语义搜索)
        """
        if not self.collection:
            return []
        results = self.collection.get(
            where={"source": {"$eq": source}},
            limit=limit,
            include=["documents", "metadatas"],
        )

        entries = []
        if results["ids"]:
            for i, mem_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                entries.append(MemoryEntry(
                    id=mem_id,
                    content=results["documents"][i] if results["documents"] else "",
                    source=metadata.get("source", "unknown"),
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    importance=float(metadata.get("importance", 0.5)),
                ))

        return entries
