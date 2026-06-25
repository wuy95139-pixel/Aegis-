"""
文件处理器
=========
从 Orchestrator 提取出的文件处理相关处理器。

包含：解析、问答、音频转录、翻译、润色、PPT生成、待办提取
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.core.agents.orchestrator_utils import get_content, extract_json, validate_file_path

logger = logging.getLogger(__name__)


class FileHandlers:
    """文件处理相关所有意图处理器"""

    def __init__(self, llm: "LLMProvider", agents: dict, translation_tool: "TranslationTool", task_store: "TaskStore", memory_manager: "MemoryManager | None" = None):
        self.llm = llm
        self.agents = agents
        self.translation_tool = translation_tool
        self.task_store = task_store
        self.memory_manager = memory_manager

    # ===================== 文件解析 =====================

    def parse(self, filepath: Optional[str]) -> dict:
        if not filepath:
            return {"status": "error", "response": "请上传一个文件让我解析。"}
        from src.core.tools.file_tools import parse_file
        validate_file_path(filepath)
        parsed = parse_file(filepath)
        return {
            "status": "success",
            "response": f"📄 **{parsed.filename}** ({parsed.file_type.value})\n"
                        f"页数: {parsed.page_count} | 字符数: {len(parsed.raw_text)}\n\n"
                        f"{parsed.raw_text[:3000]}"
        }

    # ===================== 文件问答 =====================

    def qa(self, question: str, filepath: Optional[str], stream_callback=None) -> dict:
        """针对文件内容回答问题"""
        if not filepath:
            return {"status": "error", "response": "请上传一个文件，我会从中找到答案。"}
        from src.core.tools.file_tools import parse_file
        validate_file_path(filepath)
        parsed = parse_file(filepath)
        content = parsed.raw_text
        if not content or not content.strip():
            return {"status": "error", "response": "文件内容为空，无法回答相关问题。"}
        if len(content) > 20000:
            content = content[:20000] + "\n\n(文件较长，已截取前 20000 字进行分析)"

        enhanced_context = ""
        if self.memory_manager:
            try:
                enhanced_context = self.memory_manager.get_full_context(
                    query=question,
                    include_profile=True,
                    include_rules=True,
                    include_project=True,
                    include_experiences=False,
                    include_preferences=True,
                    include_behavioral=False,
                )
            except Exception as e:
                logger.debug(f"Enhanced context for file QA failed (non-fatal): {e}")

        system_prompt = """你是一个文档问答助手。根据以下文件内容，回答用户的问题。

重要规则:
- 只回答用户问的问题，不要展示文件全部内容
- 如果文件中确实提到了相关信息，请准确引用
- 如果文件中没有提到，请明确说"文件中没有提到相关信息"
- 回答简洁准确，不要添加文件没说的内容"""

        messages = [{"role": "system", "content": system_prompt}]
        if enhanced_context:
            messages.append({"role": "system", "content": f"[用户上下文]\n{enhanced_context}"})

        messages.append({"role": "user", "content": f"""文件内容:
---
{content}
---

用户问题: {question}

请回答:"""})

        try:
            if stream_callback:
                full = ""
                for chunk in self.llm.stream_chat(messages=messages, temperature=0.3, max_tokens=2000):
                    stream_callback(chunk)
                    full += chunk
                response_content = full
            else:
                response = self.llm.chat(messages=messages, temperature=0.3, max_tokens=2000)
                response_content = response["content"]
            return {"status": "success", "response": response_content}
        except Exception as e:
            logger.error(f"File QA failed: {e}")
            return {"status": "error", "response": f"回答问题时出错: {str(e)}"}

    # ===================== 音频转录 =====================

    def audio_transcribe(self, filepath: Optional[str]) -> dict:
        """语音转录：说话人分离 + ASR + 标点恢复"""
        if not filepath:
            return {"status": "error", "response": "请上传一个音频文件，我会帮你转写成文字。"}

        ext = os.path.splitext(filepath)[1].lower()
        allowed = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
        if ext not in allowed:
            return {"status": "error", "response": f"不支持的音频格式 ({ext})。支持: {', '.join(sorted(allowed))}"}

        try:
            from src.core.tools.audio_tools import transcribe_audio, format_transcript_for_display
            result = transcribe_audio(filepath)
            display = format_transcript_for_display(result)
            return {"status": "success", "response": display, "raw_result": result}
        except ImportError:
            return {"status": "error", "response": "语音转录模块未安装。请确保 speech_pipeline1 已配置。"}
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {"status": "error", "response": f"语音转录失败: {str(e)}"}

    # ===================== 翻译 / 润色 / PPT =====================

    def translate(self, msg: str, filepath: Optional[str], params: dict, stream_callback=None) -> dict:
        target_lang = params.get("target_lang", "zh-CN")
        content = get_content(filepath, msg, params.get("text_content"))
        if not content:
            return {"status": "error", "response": "请上传文件或粘贴要翻译的文本。"}

        result = self.translation_tool.translate(content, target_lang=target_lang)
        return {
            "status": "success",
            "response": f"🌐 翻译完成 ({target_lang})\n\n{result.translated_text}"
        }

    def polish(self, msg: str, filepath: Optional[str], params: dict, stream_callback=None) -> dict:
        style = params.get("style", "professional")
        content = get_content(filepath, msg, params.get("text_content"))
        if not content:
            return {"status": "error", "response": "请上传文件或粘贴要润色的文本。"}

        polished = self.translation_tool.polish_text(content, style=style)
        return {
            "status": "success",
            "response": f"✨ 润色完成 (风格: {style})\n\n{polished}"
        }

    def generate_ppt(self, msg: str, filepath: Optional[str], params: dict, stream_callback=None) -> dict:
        content = get_content(filepath, msg, params.get("text_content"))
        if not content:
            return {"status": "error", "response": "请上传文件或粘贴要生成PPT的内容。"}

        title = params.get("title", "演示文稿")

        prompt = f"""请将以下内容分解为 PPT 幻灯片结构，每页含标题和3-5个要点。
输出 JSON: [{{"title": "页面标题", "bullets": ["要点1", "要点2"]}}, ...]

内容:
---
{content[:5000]}
---"""

        resp = self.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0.5)
        try:
            raw = extract_json(resp["content"])
            slides = json.loads(raw)
        except json.JSONDecodeError:
            slides = [{"title": title, "bullets": [line.strip() for line in content.split("\n")[:8] if line.strip()]}]

        from src.core.tools.file_tools import generate_pptx
        out = f"./output/{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        generate_pptx(title, slides, out)

        preview = "\n".join(f"**{s['title']}**\n" + "\n".join(f"- {b}" for b in s['bullets']) for s in slides[:5])
        return {"status": "success", "response": f"📊 PPT 已生成 ({len(slides)} 页)\n文件: {out}\n\n{preview}"}

    # ===================== 文件待办提取 =====================

    def extract_todos(self, filepath: Optional[str], params: dict) -> dict:
        """上传文件 → 提取待办 → 自动分派 → 创建提醒"""
        if not filepath:
            return {"status": "error", "response": "请上传一个文件（会议纪要、工作安排等），我来帮你提取待办事项并分派。"}

        from src.core.tools.file_tools import parse_file
        parsed = parse_file(filepath)

        file_result = self.agents["file_processor"].execute({
            "filepath": filepath,
            "action": "parse",
        })

        todos = file_result.get("extracted_todos", [])
        if not todos:
            return {
                "status": "success",
                "response": f"📄 **{parsed.filename}** 已解析（{len(parsed.raw_text)} 字符），但未检测到明确的待办事项。\n\n"
                           f"内容预览:\n{parsed.raw_text[:800]}..."
            }

        dispatch_result = self.agents["task_dispatcher"].execute({
            "todos": todos,
            "source": Path(filepath).name,
            "auto_assign": True,
        })

        assigned = dispatch_result.get("assigned_todos", [])
        unassigned = dispatch_result.get("unassigned", [])

        reminder_count = 0
        if assigned:
            ids = self.agents["reminder_agent"].create_reminders_from_todos(assigned)
            reminder_count = len(ids)

        all_extracted = assigned + unassigned
        if all_extracted:
            task_dicts = []
            for t in all_extracted:
                if hasattr(t, 'title'):
                    task_dicts.append({
                        "title": t.title,
                        "description": getattr(t, 'description', '') or "",
                        "priority": getattr(t, 'priority', 'medium'),
                        "deadline": t.deadline.isoformat() if hasattr(t, 'deadline') and t.deadline else None,
                    })
                else:
                    task_dicts.append(t)
            self.task_store.add_task_group(
                tasks=task_dicts,
                group_name=f"文件提取: {Path(filepath).name}",
                context=f"从文件自动提取: {Path(filepath).name}",
                set_active=True,
            )
            logger.info(f"Synced {len(task_dicts)} tasks to TaskStore from file extraction")

        self.agents["memory_agent"].execute({
            "operation": "store",
            "content": f"文件待办提取: {Path(filepath).name}, 提取了 {len(todos)} 个待办, 分派了 {len(assigned)} 个",
            "source": f"file_extract_todos:{Path(filepath).name}",
            "tags": ["file_processing", "todos", "dispatch"],
        })

        parts = [
            f"## 📋 文件待办提取完成",
            f"",
            f"**文件**: {parsed.filename} ({parsed.file_type.value})",
            f"**提取待办**: {len(todos)} 个 | **已分派**: {len(assigned)} 个 | **提醒已创建**: {reminder_count} 个",
            f"",
        ]

        if assigned:
            parts.append("### ✅ 已分派任务")
            for i, t in enumerate(assigned, 1):
                assignee = getattr(t, 'assignee', None)
                person = assignee.name if assignee else "未分配"
                deadline = t.deadline.strftime("%m-%d %H:%M") if t.deadline else "待定"
                parts.append(f"{i}. **{t.title}** → {person} (截止: {deadline})")

        if unassigned:
            parts.append(f"\n### ⚠️ 未分派 ({len(unassigned)})")
            for i, t in enumerate(unassigned, 1):
                parts.append(f"{i}. {t.get('title', str(t))}")

        return {"status": "success", "response": "\n".join(parts)}
