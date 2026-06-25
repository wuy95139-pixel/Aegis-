"""
翻译工具
=======
支持多语言翻译，默认使用 LLM 进行翻译 (可切换为专用翻译 API)。

设计决策：
  - 默认用 LLM 翻译 (质量高，术语可控)
  - 支持通过 MCP 接入专业翻译引擎 (DeepL / Google Translate)
  - 翻译请求和结果使用 TranslationRequest / TranslationResult 模型
"""

import logging
from typing import Optional

from src.models.schemas import TranslationRequest, TranslationResult

logger = logging.getLogger(__name__)


class TranslationTool:
    """
    翻译工具

    使用示例:
        tool = TranslationTool(llm_provider)
        result = tool.translate("Hello world", target_lang="zh-CN")
    """

    def __init__(self, llm_provider):
        """
        Args:
            llm_provider: LLMProvider 实例
        """
        self.llm = llm_provider

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "zh-CN",
        glossary: Optional[dict] = None,
    ) -> TranslationResult:
        """
        翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言 (auto=自动检测)
            target_lang: 目标语言
            glossary: 术语表 {"AI": "人工智能", ...}

        Returns:
            TranslationResult
        """
        # 构建翻译 prompt
        glossary_text = ""
        if glossary:
            glossary_text = "\n术语表:\n" + "\n".join(
                f"- {k} → {v}" for k, v in glossary.items()
            )

        prompt = f"""请将以下文本从{source_lang}翻译为{target_lang}。
{glossary_text}

翻译要求：
1. 保持原文格式和段落结构
2. 专业术语使用术语表中的译法
3. 保持语气和风格一致

原文：
---
{text}
---

翻译结果："""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 低温度保证翻译一致性
            )

            translated = response["content"].strip()
            logger.info(f"Translated: {len(text)} → {len(translated)} chars, {source_lang}→{target_lang}")

            return TranslationResult(
                translated_text=translated,
                source_lang_detected=source_lang if source_lang != "auto" else "en",
                target_lang=target_lang,
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                translated_text=f"[翻译失败: {e}]",
                source_lang_detected=source_lang,
                target_lang=target_lang,
                confidence=0.0,
            )

    def polish_text(
        self,
        text: str,
        style: str = "professional",
        target_length: Optional[str] = None,
    ) -> str:
        """
        文案润色

        Args:
            text: 原始文本（超过 8000 字符会被截断）
            style: 润色风格 (professional / casual / academic / concise)
            target_length: 目标长度描述，如 "缩减到100字" / "扩展到500字"

        Returns:
            润色后的文本
        """
        MAX_INPUT = 8000
        truncated = len(text) > MAX_INPUT
        input_text = text[:MAX_INPUT]

        style_prompts = {
            "professional": "使用专业、正式的商务语言风格",
            "casual": "使用轻松、口语化的风格",
            "academic": "使用严谨的学术论文风格",
            "concise": "精简到最核心的内容，去除冗余",
        }

        style_instruction = style_prompts.get(style, style_prompts["professional"])
        length_instruction = f"\n长度要求：{target_length}" if target_length else ""

        prompt = f"""请对以下文本进行润色。

风格要求：{style_instruction}{length_instruction}

润色原则：
1. 保持原意不变
2. 修正语法错误和不通顺的句子
3. 优化表达，提升可读性

原文：
---
{input_text}
---

润色结果："""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            result = response["content"].strip()
            if truncated:
                result = f"[原文共 {len(text)} 字符，仅润色前 {MAX_INPUT} 字符]\n{result}"
            return result
        except Exception as e:
            logger.error(f"Text polishing failed: {e}")
            return f"[润色失败: {e}]"
