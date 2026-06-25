"""
研究报告工作流
=============
预定义的研究分析流程：搜索 → 分析 → 生成报告 → 存储记忆

可扩展点:
  - 定时研究：设置关键词持续监控
  - 对比分析：多个主题的横向对比
  - 深度研究：迭代搜索 (用初步搜索结果提炼新的搜索词)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def run_research_workflow(
    orchestrator,
    topic: str,
    sources: Optional[list] = None,
    max_results: int = 10,
    include_swot: bool = True,
    save_to_file: Optional[str] = None,
) -> dict:
    """
    运行研究分析工作流

    流程:
      1. MemoryAgent: 检索是否有相关历史研究
      2. ResearchAgent: 搜索 + 分析 + 生成报告
      3. MemoryAgent: 存储报告摘要到长期记忆
      4. (可选) FileProcessorAgent: 生成 PDF/PPT 报告

    Args:
        orchestrator: Orchestrator 实例
        topic: 研究主题
        sources: 搜索源列表，默认 ["web", "news"]
        max_results: 每个源最大返回数
        include_swot: 是否包含 SWOT 分析
        save_to_file: 报告保存路径 (可选)

    Returns:
        工作流执行结果
    """
    if sources is None:
        sources = ["web", "news"]

    logger.info(f"Starting research workflow: topic='{topic}'")

    # Step 1: 检索相关历史研究
    memory_result = orchestrator.agents["memory_agent"].execute({
        "operation": "retrieve",
        "query": topic,
        "top_k": 3,
        "tags": ["research", "report"],
    })

    # Step 2: 执行研究
    research_result = orchestrator.agents["research_agent"].execute({
        "topic": topic,
        "sources": sources,
        "max_results": max_results,
        "include_swot": include_swot,
        "format": "markdown",
    })

    if research_result["status"] != "success":
        return {"status": "error", "step": "research", "error": research_result}

    # Step 3: 存储报告
    orchestrator.agents["memory_agent"].execute({
        "operation": "store",
        "content": f"研究报告: {topic}\n摘要: {research_result.get('markdown', '')[:800]}",
        "source": f"research:{topic}",
        "tags": ["research", "report"],
    })

    # Step 4: (可选) 保存报告文件
    if save_to_file:
        markdown_text = research_result.get("markdown", "")
        with open(save_to_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        logger.info(f"Report saved to: {save_to_file}")

    return {
        "status": "success",
        "topic": topic,
        "report_markdown": research_result.get("markdown", ""),
        "report": research_result.get("report"),
        "sources_count": research_result.get("sources_count", 0),
        "previous_research": len(memory_result.get("relevant_memories", [])),
        "saved_to": save_to_file,
    }
