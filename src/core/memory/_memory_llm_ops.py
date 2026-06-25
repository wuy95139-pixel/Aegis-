"""
LLM-backed memory operations
=============================
Key-point extraction, conversation classification, context synthesis,
and text summarization — pure functions that operate on LLM + text.

Extracted from memory_manager.py.
"""

import json
import logging
from typing import List, Dict, Optional

from src.core.memory.types import MemoryType, get_type_rule

logger = logging.getLogger(__name__)


def extract_key_points(llm, text: str, memory_type: MemoryType) -> List[str]:
    """Use LLM to extract key information points from text.

    Applies type-specific extraction strategies based on memory type rules.
    """
    if llm is None or len(text) < 20:
        return [text] if text else []

    type_rules = get_type_rule(memory_type)
    when_to_save = type_rules.get("when_to_save", [])
    focus_hint = "；".join(when_to_save) if when_to_save else (
        "user preferences, decisions, project info, contacts, important dates"
    )

    prompt = f"""Extract 3-5 key information points from the text below. Each point as one sentence.

Memory type: {memory_type.value}
Focus on: {focus_hint}

Note:
- Do NOT extract information derivable from code/git
- Do NOT extract temporary/transient information
- Only extract content with cross-session reuse value

Text:
---
{text[:2000]}
---

One point per line, no numbering."""

    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        points = [
            line.strip("-• ").strip()
            for line in response["content"].strip().split("\n")
            if line.strip()
        ]
        return points if points else [text[:200]]
    except Exception as e:
        logger.warning(f"Key point extraction failed: {e}")
        return [text[:200]]


def classify_and_extract(llm, conversation_text: str) -> Dict[str, List[Dict]]:
    """Classify conversation into four memory categories using LLM.

    Returns:
        {"user": [...], "feedback": [...], "project": [...], "reference": [...]}
    """
    prompt = f"""Analyze the conversation below. Extract information worth long-term retention,
classified into four categories:

1. **user** — user role, preferences, knowledge level, tech stack
2. **feedback** — feedback on working style ("don't X", "do more Y"), collaboration preferences
3. **project** — project background, constraints, deadlines, decision rationale
4. **reference** — external system references (URLs, tool names, platform names, etc.)

For each item provide:
- content: the information (one sentence)
- importance: 0-1 (0.5 normal, 0.7 important, 0.9 critical)
- tags: relevant tags

Do NOT extract:
- Code patterns, debugging processes
- Temporary task status
- Information derivable from git/code files

Conversation:
---
{conversation_text[:4000]}
---

Return JSON only (no other text):
{{"user": [{{"content": "...", "importance": 0.7, "tags": ["..."]}}], "feedback": [...], "project": [...], "reference": [...]}}"""

    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        content = response["content"].strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Classify and extract failed: {e}")
        return {"user": [], "feedback": [], "project": [], "reference": []}


def synthesize_context(llm, query: str, retrieval_result: dict) -> str:
    """Use LLM to synthesize retrieval results into coherent context text."""
    if llm is None:
        return retrieval_result.get("combined_context", "")

    memory_texts = []
    for entry in retrieval_result.get("file_memories", []):
        fm = entry["frontmatter"]
        type_str = fm.type.value if hasattr(fm.type, 'value') else fm.type
        memory_texts.append(f"[{type_str}] {fm.name}: {entry['content'][:300]}")

    for mem in retrieval_result.get("relevant_memories", [])[:5]:
        memory_texts.append(f"[{mem.source}] {mem.content[:300]}")

    if not memory_texts:
        return ""

    memories_text = "\n".join(f"- {t}" for t in memory_texts)

    prompt = f"""User is asking: "{query}"

Given the relevant historical memories below, generate a concise context background (2-3 sentences):

{memories_text}

Context background:"""

    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        return response["content"].strip()
    except Exception as e:
        logger.warning(f"Context synthesis failed: {e}")
        return f"Related historical information:\n{memories_text}"


def summarize_text(llm, text: str) -> str:
    """Use LLM to summarize a text in 3-5 sentences."""
    if llm is None:
        return ""

    prompt = f"""Summarize the key content of the following conversation in 3-5 sentences:

---
{text[:4000]}
---

Summary:"""

    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response["content"].strip()
    except Exception as e:
        logger.warning(f"Text summarization failed: {e}")
        return ""
