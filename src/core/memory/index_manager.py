"""
MEMORY.md 索引管理器
===================
管理中央记忆索引文件，提供结构化读写能力。

设计理念（来自 Claude Code 记忆系统）：
  - MEMORY.md 是索引而非记忆本身 — 每行一个条目
  - 每个条目：- [Title](file.md) — 一行摘要（~150 字符以内）
  - 按记忆类型分节（## User, ## Feedback, ## Project, ## Reference）
  - 始终保持在会话上下文中加载（前 200 行）

索引条目格式：
  - [记忆名称](类型目录/文件名.md) — 一行描述

职责：
  1. 在添加/删除记忆时自动同步索引
  2. 保持索引在 200 行以内（旧条目被修剪）
  3. 提供结构化读取用于会话上下文注入
  4. 支持索引重建和完整性校验
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.core.memory.types import MemoryType

logger = logging.getLogger(__name__)

# 索引条目正则: - [name](path) — description
ENTRY_PATTERN = re.compile(r"^- \[(.+?)\]\((.+?)\)\s*[—\-]\s*(.+)$")

# 索引最大行数
MAX_INDEX_LINES = 200


class IndexManager:
    """
    MEMORY.md 索引管理器

    使用示例:
        im = IndexManager("./data/memory")
        im.add_entry("user_role", MemoryType.USER, "用户是后端工程师，偏好简洁回答")
        entries = im.get_entries_by_type(MemoryType.USER)
        im.rebuild(file_store)
    """

    def __init__(self, base_dir: str = "./data/memory"):
        self.base_dir = Path(base_dir).resolve()
        self.index_path = self.base_dir / "MEMORY.md"

    # ==================== 条目管理 ====================

    def add_entry(
        self,
        name: str,
        memory_type: MemoryType,
        description: str,
        max_description_len: int = 100,
    ) -> bool:
        """
        向 MEMORY.md 添加或更新一条索引

        Args:
            name: 记忆名称
            memory_type: 记忆类型
            description: 一行描述
            max_description_len: 描述最大长度（超出截断）

        Returns:
            是否成功
        """
        # 清理描述
        desc = description.strip().replace("\n", " ")
        if len(desc) > max_description_len:
            desc = desc[:max_description_len - 3] + "..."

        entry_line = f"- [{name}]({memory_type.value}/{name}.md) — {desc}"
        section_header = f"## {memory_type.value.title()}"

        # 读取现有内容
        if self.index_path.exists():
            lines = self.index_path.read_text(encoding="utf-8").split("\n")
        else:
            lines = self._default_header()

        # 查找并替换或插入
        existing_idx = self._find_entry_line(lines, name)
        if existing_idx is not None:
            lines[existing_idx] = entry_line
        else:
            lines = self._insert_into_section(lines, section_header, entry_line)

        # 修剪过长内容
        lines = self._trim_if_needed(lines)

        self.index_path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug(f"Index entry: {name} → {memory_type.value}")
        return True

    def remove_entry(self, name: str) -> bool:
        """从索引中移除条目"""
        if not self.index_path.exists():
            return False

        lines = self.index_path.read_text(encoding="utf-8").split("\n")
        idx = self._find_entry_line(lines, name)
        if idx is None:
            return False

        del lines[idx]
        self.index_path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug(f"Index entry removed: {name}")
        return True

    def update_description(self, name: str, new_description: str) -> bool:
        """更新条目的描述部分"""
        if not self.index_path.exists():
            return False

        lines = self.index_path.read_text(encoding="utf-8").split("\n")
        idx = self._find_entry_line(lines, name)
        if idx is None:
            return False

        # 重建条目行
        match = ENTRY_PATTERN.match(lines[idx])
        if not match:
            return False

        desc = new_description.strip().replace("\n", " ")
        if len(desc) > 100:
            desc = desc[:97] + "..."

        lines[idx] = f"- [{match.group(1)}]({match.group(2)}) — {desc}"
        self.index_path.write_text("\n".join(lines), encoding="utf-8")
        return True

    # ==================== 读取操作 ====================

    def get_all_entries(self) -> Dict[str, List[Dict[str, str]]]:
        """
        读取完整索引，按类型分组

        Returns:
            {
                "user": [{"name": ..., "file": ..., "description": ...}, ...],
                "feedback": [...], "project": [...], "reference": [...]
            }
        """
        if not self.index_path.exists():
            return {}

        content = self.index_path.read_text(encoding="utf-8")
        return self._parse_index(content)

    def get_entries_by_type(self, memory_type: MemoryType) -> List[Dict[str, str]]:
        """获取指定类型的索引条目"""
        all_entries = self.get_all_entries()
        return all_entries.get(memory_type.value, [])

    def get_context_string(self, max_entries_per_type: int = 10) -> str:
        """
        获取用于注入 LLM 上下文的索引摘要字符串

        Args:
            max_entries_per_type: 每种类型最多返回条数

        Returns:
            格式化的索引文本
        """
        entries = self.get_all_entries()
        if not entries:
            return ""

        parts = ["## 记忆索引 (Memory Index)"]
        for type_name in ["user", "feedback", "project", "reference"]:
            type_entries = entries.get(type_name, [])[:max_entries_per_type]
            if not type_entries:
                continue
            parts.append(f"\n### {type_name.title()}")
            for e in type_entries:
                parts.append(f"- {e['name']}: {e['description']}")

        return "\n".join(parts)

    def find_by_keyword(self, keyword: str) -> List[Dict[str, str]]:
        """在索引中搜索关键词"""
        entries = self.get_all_entries()
        results = []
        keyword_lower = keyword.lower()

        for type_name, type_entries in entries.items():
            for entry in type_entries:
                searchable = (
                    entry.get("name", "").lower() + " " +
                    entry.get("description", "").lower()
                )
                if keyword_lower in searchable:
                    entry["type"] = type_name
                    results.append(entry)

        return results

    # ==================== 维护操作 ====================

    def rebuild(self, file_store):
        """
        从 FileStore 完全重建索引

        Args:
            file_store: FileStore 实例
        """
        from src.core.memory.file_store import FileStore

        lines = [
            "# Memory Index",
            "",
            f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}，请勿手动编辑。",
            f"> 每行不超过 150 字符，总行数不超过 {MAX_INDEX_LINES} 行。",
            "",
        ]

        for mt in MemoryType:
            entries = file_store.list_by_type(mt, limit=100)
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

        self.index_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Index rebuilt: {sum(1 for l in lines if l.startswith('- ['))} entries")

    def verify_integrity(self, file_store) -> Tuple[int, int]:
        """
        验证索引完整性

        检查是否有孤立条目（索引中有但文件不存在）
        或未索引文件（文件存在但索引中没有）

        Returns:
            (孤立条目数, 未索引文件数)
        """
        from src.core.memory.file_store import FileStore

        indexed_entries = self.get_all_entries()
        orphan_count = 0
        unindexed_count = 0

        # 收集索引中的所有文件路径
        indexed_files = set()
        for type_name, entries in indexed_entries.items():
            for entry in entries:
                file_path = self.base_dir / entry["file"]
                if file_path.exists():
                    indexed_files.add(str(file_path.resolve()))
                else:
                    orphan_count += 1
                    logger.warning(f"Orphan index entry: {entry['file']}")

        # 收集所有实际存在的文件
        actual_files = set()
        for mt in MemoryType:
            type_dir = self.base_dir / mt.value
            if type_dir.exists():
                for f in type_dir.glob("*.md"):
                    actual_files.add(str(f.resolve()))

        unindexed_files = actual_files - indexed_files
        unindexed_count = len(unindexed_files)
        for f in unindexed_files:
            logger.warning(f"Unindexed file: {f}")

        return orphan_count, unindexed_count

    def get_stats(self) -> Dict:
        """获取索引统计"""
        entries = self.get_all_entries()
        total = sum(len(v) for v in entries.values())
        return {
            "total_entries": total,
            "by_type": {k: len(v) for k, v in entries.items()},
            "file_path": str(self.index_path),
            "file_exists": self.index_path.exists(),
            "under_limit": total <= MAX_INDEX_LINES,
        }

    # ==================== 内部方法 ====================

    def _find_entry_line(self, lines: List[str], name: str) -> Optional[int]:
        """查找指定名称的条目行号（精确匹配 [name] 避免子串误匹配）"""
        escaped = re.escape(name)
        pattern = re.compile(rf"^- \[{escaped}\]\(")
        for i, line in enumerate(lines):
            if pattern.search(line):
                return i
        return None

    def _insert_into_section(
        self, lines: List[str], section_header: str, entry_line: str
    ) -> List[str]:
        """将条目插入到对应节"""
        # 查找节标题
        section_idx = None
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                section_idx = i
                break

        if section_idx is not None:
            # 找到节的结束位置（下一个 ## 或文件末尾）
            insert_at = len(lines)
            for j in range(section_idx + 1, len(lines)):
                if lines[j].startswith("## "):
                    insert_at = j
                    break
            lines.insert(insert_at, entry_line)
        else:
            # 节不存在，追加到末尾
            lines.append("")
            lines.append(section_header)
            lines.append(entry_line)

        return lines

    def _trim_if_needed(self, lines: List[str]) -> List[str]:
        """如果超过行数限制，修剪旧条目（保留节结构）"""
        if len(lines) <= MAX_INDEX_LINES:
            return lines

        # 收集各节及其条目
        result = []
        current_section_lines = []
        sections = []  # [(section_name, [lines])]

        for line in lines:
            if line.startswith("# ") or (line.startswith(">") and not line.startswith("- [")):
                result.append(line)
            elif line.startswith("## "):
                if current_section_lines:
                    sections.append((current_section_lines[0], current_section_lines))
                current_section_lines = [line]
            else:
                current_section_lines.append(line)
        if current_section_lines:
            sections.append((current_section_lines[0], current_section_lines))

        # 计算可保留的条目数
        overhead = len(result)
        for _, slines in sections:
            overhead += 1  # section header
        available = MAX_INDEX_LINES - overhead
        per_section = max(1, available // max(len(sections), 1))

        for section_header, slines in sections:
            result.append(section_header)
            entries = [l for l in slines[1:] if l.startswith("- [")]
            non_entries = [l for l in slines[1:] if not l.startswith("- [")]
            result.extend(non_entries)
            result.extend(entries[-per_section:])

        logger.info(f"Index trimmed to {len(result)} lines")
        return result

    def _default_header(self) -> List[str]:
        """默认索引头部"""
        return [
            "# Memory Index",
            "",
            "> 自动生成，请勿手动编辑。",
            "",
        ]

    def _parse_index(self, content: str) -> Dict[str, List[Dict[str, str]]]:
        """解析索引内容"""
        index: Dict[str, List[Dict[str, str]]] = {}
        current_section = None

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## "):
                current_section = stripped[3:].strip().lower()
                index.setdefault(current_section, [])
            elif stripped.startswith("- [") and current_section:
                match = ENTRY_PATTERN.match(stripped)
                if match:
                    index[current_section].append({
                        "name": match.group(1),
                        "file": match.group(2),
                        "description": match.group(3),
                    })

        return index
