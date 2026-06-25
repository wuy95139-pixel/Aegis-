"""
研究处理器
=========
处理研究分析意图：搜索网络 + LLM 生成回答或完整研究报告。
"""

import logging
from typing import Callable, Dict, Any, Optional

from src.core.tools.search_tools import SearchEngine

logger = logging.getLogger(__name__)


class ResearchHandlers:
    """研究分析处理器"""

    def __init__(self, llm: "LLMProvider", agents: dict):
        self.llm = llm
        self.agents = agents

    def research(self, params: dict, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """执行研究查询（quick 模式直接回答，full 模式生成报告）"""
        topic = params.get("topic", "")
        if not topic:
            return {"status": "error", "response": "请告诉我你想了解什么。"}

        report_mode = params.get("report_mode", "quick")

        se = SearchEngine()
        sources = []
        for src_type in ["web", "news"]:
            sources.extend(se.search(query=topic, source=src_type, max_results=8, language="zh-CN"))

        if report_mode == "full":
            return self._full_report(topic, sources)

        return self._quick_answer(topic, sources, stream_callback=stream_callback)

    def _full_report(self, topic: str, sources: list) -> Dict[str, Any]:
        result = self.agents["research_agent"].execute({
            "topic": topic,
            "sources": ["web", "news"],
            "max_results": 10,
            "include_swot": True,
            "format": "markdown",
        })
        if result["status"] != "success":
            return {"status": "error", "response": "研究分析失败，请稍后重试。"}

        self.agents["memory_agent"].execute({
            "operation": "store",
            "content": f"研究报告: {topic}",
            "source": f"research:{topic}",
            "tags": ["research", "report"],
        })
        return {"status": "success", "response": result.get("markdown", "无内容")}

    def _quick_answer(self, topic: str, sources: list, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        if not sources:
            prompt = f"请简洁回答以下问题（用中文，不超过500字）：\n\n{topic}"
        else:
            sources_text = "\n\n".join(
                f"[{i+1}] {s.title}\n{s.snippet}\n{s.url}"
                for i, s in enumerate(sources[:8])
            )
            prompt = f"""根据以下搜索资料，简洁回答用户的问题（用中文，不超过500字）。直接给核心信息即可，不需要报告格式。

搜索资料：
{sources_text[:6000]}

问题：{topic}

要求：简洁、有信息量、列出关键数据和事实。如果有不同观点可以简要提及。"""

        # 流式输出支持
        if stream_callback:
            full = ""
            try:
                for chunk in self.llm.stream_chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=1000,
                ):
                    stream_callback(chunk)
                    full += chunk
            except Exception:
                # 流式失败，回退到非流式
                resp = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=1000,
                )
                full = resp["content"]
                stream_callback(full)
            resp_content = full
        else:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            resp_content = resp["content"]

        keywords = topic[:50]
        self.agents["memory_agent"].execute({
            "operation": "store",
            "content": f"搜索问答: {keywords}, 答案摘要: {resp_content[:200]}",
            "source": f"research_quick:{keywords}",
            "tags": ["research", "quick"],
        })

        answer = resp_content
        if sources:
            answer += "\n\n---\n**参考来源**：\n" + "\n".join(
                f"- [{s.title}]({s.url})" for s in sources[:5]
            )

        return {"status": "success", "response": answer}
