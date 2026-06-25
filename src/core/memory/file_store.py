"""
文件记忆存储
============
基于 Markdown + YAML Frontmatter 的文件持久化层。

设计理念（来自 Claude Code 记忆系统）：
  - 每个记忆一个 .md 文件，人类可读、Git 友好
  - YAML frontmatter 存储结构化元数据
  - 文件系统即数据库，无需额外服务
  - 按记忆类型分目录组织

与 Aegis 原有系统的关系：
  - FileStore 是 SOURCE OF TRUTH（数据源头）
  - ChromaDB 是 SEARCH INDEX（搜索加速）
  - 两者通过 MemoryManager 保持同步

目录结构：
  memory/
    user/         — 用户记忆
    feedback/     — 反馈记忆
    project/      — 项目记忆
    reference/    — 参考记忆
    MEMORY.md     — 中央索引
"""

import os
import re
import uuid
import logging
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import yaml

from src.core.memory.types import (
    MemoryType, MemoryFrontmatter, is_worth_remembering,
)

logger = logging.getLogger(__name__)

# Frontmatter 正则：匹配 --- ... ---
FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class FileStore:
    """
    基于 Markdown 文件的记忆存储

    使用示例:
        fs = FileStore(base_dir="./data/memory")
        fs.save(MemoryFrontmatter(name="user_role", type="user"), "用户是后端工程师")
        results = fs.search_by_type(MemoryType.USER)
        entry = fs.get("user_role")
    """

    def __init__(self, base_dir: str = "./data/memory"):
        """
        Args:
            base_dir: 记忆文件根目录
        """
        self.base_dir = Path(base_dir).resolve()
        self._index_lock = threading.Lock()
        self._ensure_dirs()

    # ==================== 目录结构 ====================

    def _ensure_dirs(self):
        """确保所有类型目录存在"""
        for mt in MemoryType:
            (self.base_dir / mt.value).mkdir(parents=True, exist_ok=True)

    def _get_path(self, memory_type: MemoryType, name: str) -> Path:
        """获取记忆文件路径"""
        # 清理文件名中的非法字符
        safe_name = re.sub(r"[<>:\"/\\|?*]", "_", name)
        return self.base_dir / memory_type.value / f"{safe_name}.md"

    def _get_index_path(self) -> Path:
        """获取 MEMORY.md 索引文件路径"""
        return self.base_dir / "MEMORY.md"

    # ==================== CRUD 操作 ====================

    def save(
        self,
        frontmatter: MemoryFrontmatter,
        content: str,
        update_index: bool = True,
    ) -> Path:
        """
        保存一条记忆到文件

        Args:
            frontmatter: Frontmatter 元数据
            content: 记忆正文（Markdown 格式）
            update_index: 是否同步更新 MEMORY.md

        Returns:
            文件路径
        """
        if frontmatter.created_at is None:
            frontmatter.created_at = datetime.now()
        frontmatter.updated_at = datetime.now()

        file_path = self._get_path(frontmatter.type, frontmatter.name)

        # 构建 frontmatter YAML
        fm_dict = frontmatter.model_dump(exclude_none=True, exclude_defaults=False)
        # 移除空列表和 None 值以保持 frontmatter 简洁
        fm_dict = {
            k: v for k, v in fm_dict.items()
            if v is not None and v != [] and v != "" and v != 0.5
        }
        # 但必须保留 type, name, description
        for key in ["type", "name", "description"]:
            if key not in fm_dict:
                fm_dict[key] = getattr(frontmatter, key)

        # 将 type enum 转为字符串
        fm_dict["type"] = fm_dict["type"].value if isinstance(fm_dict["type"], MemoryType) else fm_dict["type"]

        # 格式化日期
        for date_key in ["created_at", "updated_at"]:
            if date_key in fm_dict and isinstance(fm_dict[date_key], datetime):
                fm_dict[date_key] = fm_dict[date_key].isoformat()

        frontmatter_yaml = yaml.dump(fm_dict, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()

        # 写入文件
        file_content = f"---\n{frontmatter_yaml}\n---\n\n{content}\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        logger.debug(f"Saved memory file: {file_path}")

        # 更新索引
        if update_index:
            self._update_index_entry(frontmatter.name, frontmatter.type, frontmatter.description)

        return file_path

    def get(self, name: str, memory_type: Optional[MemoryType] = None) -> Optional[Dict[str, Any]]:
        """
        读取一条记忆

        Args:
            name: 记忆名称
            memory_type: 记忆类型，None 则搜索所有类型

        Returns:
            {"frontmatter": MemoryFrontmatter, "content": str} 或 None
        """
        if memory_type:
            types_to_search = [memory_type]
        else:
            types_to_search = list(MemoryType)

        for mt in types_to_search:
            file_path = self._get_path(mt, name)
            if file_path.exists():
                return self._parse_file(file_path)

        return None

    def delete(self, name: str, memory_type: MemoryType, update_index: bool = True) -> bool:
        """
        删除一条记忆

        Returns:
            是否成功删除
        """
        file_path = self._get_path(memory_type, name)
        if not file_path.exists():
            return False

        file_path.unlink()
        logger.debug(f"Deleted memory file: {file_path}")

        if update_index:
            self._remove_index_entry(name)

        return True

    def update(
        self,
        name: str,
        memory_type: MemoryType,
        content: Optional[str] = None,
        frontmatter_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        更新一条记忆

        Args:
            name: 记忆名称
            memory_type: 记忆类型
            content: 新内容（None 表示不修改）
            frontmatter_updates: 要更新的 frontmatter 字段

        Returns:
            是否成功
        """
        entry = self.get(name, memory_type)
        if entry is None:
            return False

        existing_fm = entry["frontmatter"]
        new_content = content if content is not None else entry["content"]

        # 合并 frontmatter 更新
        if frontmatter_updates:
            fm_dict = existing_fm.model_dump()
            fm_dict.update(frontmatter_updates)
            existing_fm = MemoryFrontmatter(**fm_dict)

        existing_fm.updated_at = datetime.now()

        self.save(existing_fm, new_content, update_index=bool(frontmatter_updates))
        return True

    # ==================== 查询操作 ====================

    def list_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        列出指定类型的所有记忆

        Returns:
            [{"frontmatter": ..., "content": ..., "file_path": ...}, ...]
        """
        type_dir = self.base_dir / memory_type.value
        if not type_dir.exists():
            return []

        entries = []
        for file_path in sorted(type_dir.glob("*.md"), key=os.path.getmtime, reverse=True)[:limit]:
            parsed = self._parse_file(file_path)
            if parsed:
                parsed["file_path"] = str(file_path)
                entries.append(parsed)

        return entries

    def search_by_tags(
        self,
        tags: List[str],
        memory_type: Optional[MemoryType] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """按标签搜索记忆"""
        types_to_search = [memory_type] if memory_type else list(MemoryType)
        results = []

        for mt in types_to_search:
            for entry in self.list_by_type(mt, limit=200):
                entry_tags = entry["frontmatter"].tags or []
                if any(t in entry_tags for t in tags):
                    results.append(entry)

        # 按更新时间排序
        results.sort(key=lambda e: str(e["frontmatter"].updated_at or ""), reverse=True)
        return results[:limit]

    def search_by_importance(
        self,
        min_importance: float = 0.7,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """按重要性阈值搜索记忆"""
        results = []
        for mt in MemoryType:
            for entry in self.list_by_type(mt, limit=200):
                if entry["frontmatter"].importance >= min_importance:
                    results.append(entry)
        results.sort(key=lambda e: e["frontmatter"].importance, reverse=True)
        return results[:limit]

    def full_text_search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        全文搜索（关键词匹配，用于无向量数据库时的降级方案）

        匹配策略（按优先级）:
          1. query 整体作为子串匹配（精确匹配，适合短关键词）
          2. query 拆分为双字词 (bigram) 匹配（适合中文长查询）
          3. query 中的单字匹配至少3个字（适合短语查询）

        Args:
            query: 搜索关键词/问题
            memory_type: 限定类型
            limit: 返回条数
        """
        types_to_search = [memory_type] if memory_type else list(MemoryType)
        query_stripped = query.strip()
        query_lower = query_stripped.lower()
        results = []

        # 生成查询的 bigram 词组（用于中文匹配）
        # 例如 "之前提到的架构是什么" → ["之前", "前提", "提到", "到的", "的架", "架构", "构是", "是什", "什么"]
        query_bigrams = set()
        for i in range(len(query_stripped) - 1):
            bigram = query_stripped[i:i+2]
            query_bigrams.add(bigram)

        # 提取查询中的有效单字（过滤标点和空格）
        query_chars = [c for c in query_stripped if c.strip() and c not in "，。！？、；：""''（）【】《》…—·,.;:!?"]
        query_char_set = set(query_chars)

        for mt in types_to_search:
            for entry in self.list_by_type(mt, limit=500):
                content = entry["content"].lower()
                fm = entry["frontmatter"]
                searchable = (
                    content + " " +
                    fm.name.lower() + " " +
                    fm.description.lower() + " " +
                    " ".join(fm.tags).lower()
                )

                # 策略1: 精确子串匹配（适合短关键词如"架构"、"微服务"）
                if query_lower in searchable:
                    results.append(entry)
                    continue

                # 策略2: Bigram 匹配（中文词级匹配）
                # 至少2个bigram匹配（绝对数），或至少30%匹配率（相对数）
                # 兼顾长查询（bigram多但单个匹配也很有意义）和短查询
                matched_bigrams = sum(1 for bg in query_bigrams if bg.lower() in searchable)
                bigram_ratio = matched_bigrams / max(len(query_bigrams), 1)
                if matched_bigrams >= 2 or bigram_ratio >= 0.3:
                    results.append(entry)
                    continue

                # 策略3: 单字匹配（至少3个字符匹配，或至少40%匹配率）
                if len(query_char_set) >= 3:
                    matched_chars = sum(1 for ch in query_char_set if ch.lower() in searchable)
                    char_ratio = matched_chars / len(query_char_set)
                    if matched_chars >= 3 or char_ratio >= 0.4:
                        results.append(entry)
                        continue

        return results[:limit]

    # ==================== 索引管理 ====================

    def _update_index_entry(self, name: str, memory_type: MemoryType, description: str):
        """更新 MEMORY.md 中的条目（线程安全）"""
        with self._index_lock:
            self._update_index_entry_locked(name, memory_type, description)

    def _update_index_entry_locked(self, name: str, memory_type: MemoryType, description: str):
        index_path = self._get_index_path()
        entry_line = f"- [{name}]({memory_type.value}/{name}.md) — {description}"

        if not index_path.exists():
            self._rebuild_index_locked()
            return

        lines = index_path.read_text(encoding="utf-8").split("\n")

        # 查找是否已存在同一条目
        existing_idx = None
        for i, line in enumerate(lines):
            if f"[{name}]" in line:
                existing_idx = i
                break

        if existing_idx is not None:
            lines[existing_idx] = entry_line
        else:
            # 插入到对应类型小节
            section_header = f"## {memory_type.value.title()}"
            inserted = False
            for i, line in enumerate(lines):
                if line.strip() == section_header:
                    # 找到对应节，插入到该节末尾（下一个节之前）
                    insert_at = i + 1
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("## "):
                            insert_at = j
                            break
                    else:
                        insert_at = len(lines)
                    lines.insert(insert_at, entry_line)
                    inserted = True
                    break

            if not inserted:
                # 没有对应节，追加新节
                lines.append(f"\n## {memory_type.value.title()}")
                lines.append(entry_line)

        # 写入，确保不超过 200 行
        if len(lines) > 200:
            # 移除最旧的条目（保留标题行）
            trimmed = []
            for line in lines:
                if line.startswith("#") or line.startswith("##"):
                    trimmed.append(line)
            # 保留最近的条目
            entry_lines = [l for l in lines if l.startswith("- [")]
            trimmed.extend(entry_lines[:180])  # 留 20 行给标题
            lines = trimmed

        index_path.write_text("\n".join(lines), encoding="utf-8")

    def _remove_index_entry(self, name: str):
        """从 MEMORY.md 移除条目（线程安全）"""
        with self._index_lock:
            index_path = self._get_index_path()
            if not index_path.exists():
                return

            lines = index_path.read_text(encoding="utf-8").split("\n")
            lines = [l for l in lines if f"[{name}]" not in l]
            index_path.write_text("\n".join(lines), encoding="utf-8")

    def _rebuild_index(self):
        """完全重建 MEMORY.md（线程安全）"""
        with self._index_lock:
            self._rebuild_index_locked()

    def _rebuild_index_locked(self):
        index_path = self._get_index_path()
        lines = [
            "# Memory Index",
            "",
            "> 自动生成，请勿手动编辑。每行不超过 150 字符。",
            "",
        ]

        for mt in MemoryType:
            entries = self.list_by_type(mt, limit=100)
            if not entries:
                continue
            lines.append(f"## {mt.value.title()}")
            for entry in entries:
                fm = entry["frontmatter"]
                desc = fm.description or ""
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                lines.append(f"- [{fm.name}]({mt.value}/{fm.name}.md) — {desc}")
            lines.append("")

        index_path.write_text("\n".join(lines), encoding="utf-8")

    def get_index_entries(self) -> Dict[str, List[Dict[str, str]]]:
        """
        读取 MEMORY.md 并返回结构化索引

        Returns:
            { "user": [{"name": ..., "file": ..., "description": ...}, ...], ... }
        """
        index_path = self._get_index_path()
        if not index_path.exists():
            return {}

        content = index_path.read_text(encoding="utf-8")
        index: Dict[str, List[Dict[str, str]]] = {}
        current_section = None

        for line in content.split("\n"):
            if line.startswith("## "):
                current_section = line[3:].strip().lower()
                index.setdefault(current_section, [])
            elif line.startswith("- [") and current_section:
                # 解析: - [name](file.md) — description
                match = re.match(r"- \[(.+?)\]\((.+?)\) — (.+)", line)
                if match:
                    index[current_section].append({
                        "name": match.group(1),
                        "file": match.group(2),
                        "description": match.group(3),
                    })

        return index

    # ==================== 工具方法 ====================

    def _parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析记忆文件为 frontmatter + content"""
        try:
            raw = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None

        fm_match = FM_PATTERN.match(raw)
        if not fm_match:
            logger.warning(f"No frontmatter found in {file_path}")
            return None

        try:
            fm_dict = yaml.safe_load(fm_match.group(1))
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML in {file_path}: {e}")
            return None

        content = raw[fm_match.end():].strip()

        # 确保 type 是枚举
        if "type" in fm_dict and isinstance(fm_dict["type"], str):
            fm_dict["type"] = MemoryType(fm_dict["type"])

        frontmatter = MemoryFrontmatter(**fm_dict)

        return {
            "frontmatter": frontmatter,
            "content": content,
        }

    def count_by_type(self) -> Dict[str, int]:
        """统计各类型记忆数量"""
        counts = {}
        for mt in MemoryType:
            type_dir = self.base_dir / mt.value
            counts[mt.value] = len(list(type_dir.glob("*.md"))) if type_dir.exists() else 0
        return counts

    def total_count(self) -> int:
        """总记忆数"""
        return sum(self.count_by_type().values())

    def get_stats(self) -> Dict[str, Any]:
        """获取文件存储统计信息"""
        counts = self.count_by_type()
        return {
            "base_dir": str(self.base_dir),
            "total_memories": sum(counts.values()),
            "by_type": counts,
            "has_index": self._get_index_path().exists(),
        }
