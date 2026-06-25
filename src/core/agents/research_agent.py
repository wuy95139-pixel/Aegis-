"""
研究分析代理 (ResearchAgent)
===========================
职责：
  1. 实时搜索新闻、热点话题
  2. 深度分析搜索结果的多个维度
  3. 生成结构化报告 (含摘要、SWOT分析、来源引用)

协作关系：
  输入: 研究主题 (ResearchQuery)
  输出: 结构化报告 (ResearchReport) → 可进一步交给 FileProcessorAgent 生成文档

可扩展点：
  - 多源数据融合：社交媒体、学术论文、专利数据库
  - 情感分析：对搜索结果进行情感判断
  - 趋势预测：基于历史数据预测趋势走向
  - 定时研究监控：设置关键词，持续监控并推送更新
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.agents.base import BaseAgent
from src.core.tools.search_tools import SearchEngine
from src.models.schemas import (
    ResearchQuery, ResearchReport, ResearchSource,
    SWOTAnalysis, ReportFormat,
)

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """研究分析代理 — 搜索、分析、生成报告"""

    role = "研究分析专家"
    goal = "实时搜索最新信息，进行深度分析并生成高质量的结构化研究报告，帮助用户快速了解任何话题"
    backstory = """
你是一位资深的研究分析师，拥有出色的信息搜集和分析能力。
你需要：
- 快速从多个来源搜索相关信息 (网页、新闻、学术)
- 对搜索结果进行深度分析，识别关键趋势和模式
- 生成 SWOT 分析，评估机会和风险
- 撰写结构清晰、引用准确的研究报告
- 区分事实和观点，标注信息来源的可信度

你的报告应该包含：
1. 执行摘要 — 300字以内的核心发现
2. 背景介绍 — 话题的背景和上下文
3. 关键发现 — 3-7个核心分析点
4. SWOT分析 — 优势、劣势、机会、威胁
5. 来源列表 — 所有引用的来源链接
6. 建议 — 基于分析的后续行动建议
"""

    def __init__(self, llm, memory=None, config=None):
        super().__init__(
            name="research_agent",
            llm=llm,
            memory=memory,
            tools=[],
            config=config,
        )
        self.search_engine = SearchEngine()

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行研究分析

        task_input 结构:
          {
            "topic": "2024年AI智能体发展趋势",
            "keywords": ["AI agent", "多智能体", "CrewAI"],
            "sources": ["web", "news"],
            "max_results": 10,
            "language": "zh-CN",
            "include_swot": true,
            "format": "markdown",
          }

        Returns:
          {
            "status": "success" | "error",
            "report": ResearchReport,
            "markdown": "...",                    # Markdown 格式的报告文本
          }
        """
        topic = task_input.get("topic", "")
        keywords = task_input.get("keywords", [])
        sources = task_input.get("sources", ["web", "news"])
        max_results = task_input.get("max_results", 10)
        language = task_input.get("language", "zh-CN")
        include_swot = task_input.get("include_swot", True)
        report_format = task_input.get("format", "markdown")

        if not topic:
            return {"status": "error", "message": "研究主题 (topic) 不能为空"}

        logger.info(f"ResearchAgent: topic='{topic}', sources={sources}")

        # Step 1: 搜索
        all_sources: List[ResearchSource] = []
        for source_type in sources:
            results = self.search_engine.search(
                query=topic,
                source=source_type,
                max_results=max_results,
                language=language,
            )
            all_sources.extend(results)

        logger.info(f"Research: found {len(all_sources)} sources")

        # Step 2: 分析并生成报告
        report = self._generate_report(
            topic=topic,
            sources=all_sources,
            keywords=keywords,
            include_swot=include_swot,
        )

        # Step 3: 格式化为 Markdown
        markdown_text = self._format_report_markdown(report)

        # Step 4: 存储到长期记忆
        self.remember(
            content=f"研究报告: {topic}, 发现{len(report.findings)}个关键点",
            source=f"research:{topic}",
            tags=["research", "report"],
        )

        return {
            "status": "success",
            "report": report,
            "markdown": markdown_text,
            "sources_count": len(all_sources),
        }

    def _generate_report(
        self,
        topic: str,
        sources: List[ResearchSource],
        keywords: List[str],
        include_swot: bool = True,
        language: str = "zh-CN",
    ) -> ResearchReport:
        """使用 LLM 生成结构化研究报告"""
        # 构建分析上下文
        sources_text = "\n\n".join(
            f"来源 {i+1}: {s.title}\nURL: {s.url}\n摘要: {s.snippet}"
            for i, s in enumerate(sources[:15])
        )

        prompt = f"""你是一位资深研究分析师。请根据以下搜索资料，生成一份关于"{topic}"的结构化研究报告。

搜索资料：
{sources_text[:8000]}

请按以下结构输出报告：

## 执行摘要
(200-300字的核心发现总结)

## 背景介绍
(话题的背景和重要性)

## 关键发现
(5-7个要点，每个要点2-3句话，基于搜索资料的事实)

## SWOT分析
- **优势 (Strengths)**:
- **劣势 (Weaknesses)**:
- **机会 (Opportunities)**:
- **威胁 (Threats)**:

## 建议
(3-5条基于分析的后续行动建议)

## 来源
(列出引用的来源)

请用{language}撰写报告。"""

        try:
            report_text = self._run_crew_task(
                description=prompt,
                expected_output="一份结构化的中文研究报告，包含执行摘要、背景介绍、关键发现、SWOT分析、建议和来源",
            )
            return self._parse_report(report_text, topic, sources)
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return ResearchReport(
                title=f"研究报告: {topic}",
                executive_summary=f"报告生成失败: {e}",
                introduction="",
                sources=sources,
            )

    def _parse_report(
        self, raw_text: str, topic: str, sources: List[ResearchSource]
    ) -> ResearchReport:
        """解析 LLM 生成的报告文本为结构化对象"""
        # TODO: 更精确的文本结构化解析
        # 当前简化处理
        lines = raw_text.split("\n")

        executive_summary = ""
        introduction = ""
        findings: List[str] = []
        swot = SWOTAnalysis()
        recommendations: List[str] = []

        current_section = ""
        for line in lines:
            line = line.strip()
            if "执行摘要" in line:
                current_section = "summary"
                continue
            elif "背景介绍" in line:
                current_section = "intro"
                continue
            elif "关键发现" in line:
                current_section = "findings"
                continue
            elif "SWOT" in line:
                current_section = "swot"
                continue
            elif "建议" in line:
                current_section = "recommendations"
                continue
            elif "来源" in line:
                current_section = "sources"
                continue

            if current_section == "summary" and line:
                executive_summary += line + "\n"
            elif current_section == "intro" and line:
                introduction += line + "\n"
            elif current_section == "findings" and line.startswith(("-", "*", "1.", "2.", "3.")):
                findings.append(line.lstrip("-* 0123456789."))
            elif current_section == "swot":
                if "优势" in line or "Strength" in line:
                    swot.strengths.append(line.split(":", 1)[-1].strip())
                elif "劣势" in line or "Weakness" in line:
                    swot.weaknesses.append(line.split(":", 1)[-1].strip())
                elif "机会" in line or "Opportun" in line:
                    swot.opportunities.append(line.split(":", 1)[-1].strip())
                elif "威胁" in line or "Threat" in line:
                    swot.threats.append(line.split(":", 1)[-1].strip())
            elif current_section == "recommendations" and line.startswith(("-", "*", "1.", "2.")):
                recommendations.append(line.lstrip("-* 0123456789."))

        return ResearchReport(
            title=f"研究报告: {topic}",
            executive_summary=executive_summary.strip(),
            introduction=introduction.strip(),
            findings=findings,
            swot=swot if any([swot.strengths, swot.weaknesses, swot.opportunities, swot.threats]) else None,
            sources=sources,
            recommendations=recommendations,
            format=ReportFormat.MARKDOWN,
        )

    def _format_report_markdown(self, report: ResearchReport) -> str:
        """将报告格式化为 Markdown"""
        md = f"# {report.title}\n\n"
        md += f"*生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M')}*\n\n"

        md += "## 执行摘要\n\n"
        md += f"{report.executive_summary}\n\n"

        if report.introduction:
            md += "## 背景介绍\n\n"
            md += f"{report.introduction}\n\n"

        if report.findings:
            md += "## 关键发现\n\n"
            for i, finding in enumerate(report.findings, 1):
                md += f"{i}. {finding}\n"
            md += "\n"

        if report.swot:
            md += "## SWOT 分析\n\n"
            md += "| 维度 | 内容 |\n"
            md += "|------|------|\n"
            for s in report.swot.strengths:
                md += f"| 优势 (S) | {s} |\n"
            for w in report.swot.weaknesses:
                md += f"| 劣势 (W) | {w} |\n"
            for o in report.swot.opportunities:
                md += f"| 机会 (O) | {o} |\n"
            for t in report.swot.threats:
                md += f"| 威胁 (T) | {t} |\n"
            md += "\n"

        if report.recommendations:
            md += "## 建议\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                md += f"{i}. {rec}\n"
            md += "\n"

        md += "## 信息来源\n\n"
        for source in report.sources[:10]:
            md += f"- [{source.title}]({source.url})\n"

        return md
