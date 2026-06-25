"""
搜索工具集
=========
实时搜索网页、新闻。使用 Tavily Search API。

设计决策：
  - Tavily SDK 官方集成
  - 搜索结果统一转为 ResearchSource 模型
  - 无 API Key 时回退到模拟数据

可扩展点：
  - 新增搜索引擎：实现 _search_xxx 方法并在 search() 中注册
  - 学术搜索：Semantic Scholar / Google Scholar API
"""

import logging
import os
from typing import List, Optional
from datetime import datetime

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from src.models.schemas import ResearchSource

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    统一搜索引擎接口（Tavily）

    使用示例:
        engine = SearchEngine(api_key="tvly-xxx")
        results = engine.search("AI 智能体 2026", source="web", max_results=5)
        news = engine.search_news("科技", max_results=5, days=3)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self._client = None
        if TavilyClient and self.api_key:
            self._client = TavilyClient(api_key=self.api_key)

    def _get_client(self):
        """延迟初始化 Tavily 客户端"""
        if self._client is None and TavilyClient and self.api_key:
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def search(
        self,
        query: str,
        source: str = "web",
        max_results: int = 10,
        language: str = "zh-CN",
    ) -> List[ResearchSource]:
        """
        执行搜索

        Args:
            query: 搜索查询
            source: 搜索源 (web / news)
            max_results: 最大返回数
            language: 语言偏好

        Returns:
            搜索结果列表
        """
        logger.info(f"Search: query='{query}', source={source}, max={max_results}")

        client = self._get_client()
        if not client:
            logger.warning("Tavily API key not configured, using mock search")
            return self._search_mock(query, source, max_results)

        try:
            topic = "news" if source == "news" else "general"
            response = client.search(
                query=query,
                topic=topic,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
            )
            return [
                ResearchSource(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    source_type=source,
                    relevance_score=r.get("score", 0.5),
                )
                for r in response.get("results", [])[:max_results]
            ]
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return self._search_mock(query, source, max_results)

    def search_news(
        self, query: str, max_results: int = 10, days: int = 7
    ) -> List[ResearchSource]:
        """
        搜索新闻

        Args:
            query: 新闻搜索词
            max_results: 最大返回数
            days: 时间范围 (最近 N 天)
        """
        client = self._get_client()
        if not client:
            return self._search_mock(query, "news", max_results)

        try:
            response = client.search(
                query=query,
                topic="news",
                search_depth="advanced",
                max_results=max_results,
                days=days,
                include_answer=False,
            )
            return [
                ResearchSource(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    source_type="news",
                    relevance_score=r.get("score", 0.5),
                )
                for r in response.get("results", [])[:max_results]
            ]
        except Exception as e:
            logger.error(f"Tavily news search failed: {e}")
            return self._search_mock(query, "news", max_results)

    def _search_mock(
        self,
        query: str,
        source: str = "web",
        max_results: int = 10,
    ) -> List[ResearchSource]:
        """模拟搜索结果 (无 API Key 时使用)"""
        return [
            ResearchSource(
                title=f"[模拟结果 {i}] 关于 '{query}' 的搜索结果 — 配置 TAVILY_API_KEY 获取真实数据",
                url=f"https://example.com/result-{i}",
                snippet=f"当前未配置 Tavily API Key，显示模拟数据。请在 .env 中设置 TAVILY_API_KEY 以启用真实搜索。",
                source_type=source,
                relevance_score=0.8 - i * 0.05,
            )
            for i in range(min(max_results, 5))
        ]

    def close(self):
        """关闭客户端"""
        self._client = None
