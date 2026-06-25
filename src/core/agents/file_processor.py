"""
文件处理代理 (FileProcessorAgent)
================================
职责：
  1. 解析用户上传的文件 (PDF/Word/PPT/图片)
  2. 根据用户要求进行翻译、PPT生成、文案润色
  3. 将处理结果传递给任务分发代理

协作关系：
  输入: 用户文件 + 处理指令
  输出: ParsedFile + 处理结果 → 传递给 TaskDispatcherAgent

可扩展点：
  - 更多文件格式支持 (Excel, ePub, HTML)
  - 批量文件处理
  - 文件版本对比
"""

import logging
from typing import Dict, Any, List, Optional

from src.core.agents.base import BaseAgent
from src.core.tools.file_tools import parse_file, generate_pptx, detect_file_type
from src.core.tools.translation_tools import TranslationTool
from src.models.schemas import ParsedFile, TranslationResult, FileType

logger = logging.getLogger(__name__)


class FileProcessorAgent(BaseAgent):
    """文件处理代理 — 解析、翻译、润色、PPT生成"""

    role = "文件处理专家"
    goal = "深度解析各类文档文件，精准提取内容，并按要求进行翻译、润色或PPT生成"
    backstory = """
你是一位资深的文档处理专家，精通各类文件格式的解析和内容提取。
你能够：
- 快速解析 PDF、Word、PPT、图片等文件，提取文本、表格和元数据
- 高质量地将内容翻译为多种语言
- 对文案进行专业润色，提升表达质量
- 从原始内容生成结构清晰的 PPT 演示文稿

你需要准确理解用户的处理需求，选择合适的工具完成任务。
在提取到内容后，你需要识别其中包含的待办事项和行动点，以便后续分派。
"""

    def __init__(self, llm, memory=None, config=None):
        # 初始化工具
        self.translation_tool = TranslationTool(llm)

        tools = []  # CrewAI tools (目前使用内部方法)
        super().__init__(
            name="file_processor",
            llm=llm,
            memory=memory,
            tools=tools,
            config=config,
        )

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行文件处理任务

        task_input 结构:
          {
            "filepath": "/path/to/file.pdf",        # 文件路径 (必需)
            "action": "translate",                   # 操作: parse / translate / polish / generate_ppt
            "target_lang": "zh-CN",                  # 翻译目标语言
            "style": "professional",                  # 润色风格
            "ppt_title": "标题",                      # PPT 标题
            "glossary": {"AI": "人工智能"},          # 翻译术语表
          }

        Returns:
          {
            "status": "success" | "error",
            "parsed_file": ParsedFile,
            "result_text": "...",
            "extracted_todos": [...],                # 提取的待办事项 (用于后续分派)
          }
        """
        filepath = task_input.get("filepath", "")
        action = task_input.get("action", "parse")

        logger.info(f"FileProcessor processing: {filepath}, action={action}")

        # Step 1: 解析文件
        parsed = parse_file(filepath)
        if parsed.file_type == FileType.UNKNOWN:
            return {"status": "error", "message": f"不支持的文件格式: {filepath}"}

        result = {
            "status": "success",
            "parsed_file": parsed,
            "result_text": parsed.raw_text,
            "extracted_todos": [],
        }

        # Step 2: 根据 action 执行处理
        if action == "translate":
            target_lang = task_input.get("target_lang", "zh-CN")
            glossary = task_input.get("glossary")

            translation = self.translation_tool.translate(
                text=parsed.raw_text,
                source_lang="auto",
                target_lang=target_lang,
                glossary=glossary,
            )
            result["result_text"] = translation.translated_text
            result["translation"] = translation

        elif action == "polish":
            style = task_input.get("style", "professional")
            polished = self.translation_tool.polish_text(
                text=parsed.raw_text,
                style=style,
            )
            result["result_text"] = polished

        elif action == "generate_ppt":
            ppt_title = task_input.get("ppt_title", parsed.filename)
            # TODO: 用 LLM 将内容分解为幻灯片结构
            slides_content = self._content_to_slides(parsed.raw_text, ppt_title)
            output_path = task_input.get("output_path", f"./output/{parsed.filename}.pptx")
            generate_pptx(ppt_title, slides_content, output_path)
            result["ppt_path"] = output_path

        # Step 3: 提取待办事项 (用于后续任务分派)
        todos = self._extract_todos(result["result_text"])
        result["extracted_todos"] = todos

        # Step 4: 存入长期记忆 (重要文件处理记录)
        self.remember(
            content=f"处理文件: {parsed.filename}, 操作: {action}, 摘要: {result['result_text'][:500]}",
            source=f"file:{parsed.filename}",
            tags=["file_processing", action],
        )

        return result

    def _extract_todos(self, text: str) -> List[Dict[str, str]]:
        """
        从文本中提取待办事项

        使用 LLM 识别文本中的行动点、任务和截止日期。

        可扩展点：
          - 识别负责人 (如 "@张三")
          - 识别截止日期 (如 "by Friday", "截止: 2024-06-01")
          - 优先级推断
        """
        if not text or len(text) < 10:
            return []

        # TODO: 使用 LLM 精确提取
        prompt = f"""从以下文本中提取所有待办事项和行动点。
对每个待办事项，提取：标题、描述、负责人(如有)、截止日期(如有)。

文本：
---
{text[:4000]}
---

以 JSON 格式输出：
[
  {{"title": "...", "description": "...", "assignee": "...", "deadline": "..."}},
  ...
]

如果没有待办事项，输出空数组 []。"""

        try:
            result_text = self._run_crew_task(
                description=prompt,
                expected_output="一个 JSON 数组，包含所有待办事项，格式为 [{title, description, assignee, deadline}]",
            )
            import json
            raw = result_text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            todos = json.loads(raw)
            logger.info(f"Extracted {len(todos)} todos from text")
            return todos
        except Exception as e:
            logger.warning(f"Todo extraction failed: {e}")
            return []

    def _content_to_slides(
        self, text: str, title: str
    ) -> List[Dict[str, Any]]:
        """
        将文本内容转为 PPT 幻灯片结构

        使用 LLM 自动分段，生成每页幻灯片的标题和要点。
        """
        # TODO: 用 LLM 将长文本分解为幻灯片
        # 当前返回简单分段
        paragraphs = text.split("\n\n")
        slides = []
        for i, p in enumerate(paragraphs[:20]):  # 最多 20 页
            if p.strip():
                slides.append({
                    "title": f"{title} ({i+1})" if i > 0 else title,
                    "bullets": [line.strip() for line in p.split("\n")[:5] if line.strip()],
                })
        return slides or [{"title": title, "bullets": ["(无内容)"]}]
