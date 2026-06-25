"""
core/tools/search_tools.py 测试
===============================
SearchEngine 的搜索、新闻搜索、mock 回退测试。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.tools.search_tools import SearchEngine
from src.models.schemas import ResearchSource


class TestSearchEngine:
    @pytest.fixture
    def engine_no_key(self, monkeypatch):
        """无 API Key 的搜索引擎 — 确保环境变量也清除。"""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        return SearchEngine(api_key="")

    def test_mock_search_on_no_key(self, engine_no_key):
        results = engine_no_key.search("AI 智能体")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, ResearchSource)

    def test_mock_search_news_on_no_key(self, engine_no_key):
        results = engine_no_key.search_news("科技", max_results=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert r.source_type == "news"

    def test_mock_search_max_results(self, engine_no_key):
        results = engine_no_key.search("query", max_results=3)
        # mock 最多返回 min(max_results, 5)
        assert len(results) <= 3

    def test_mock_search_web_source(self, engine_no_key):
        results = engine_no_key.search("test", source="web")
        for r in results:
            assert r.source_type == "web"

    def test_mock_search_news_source(self, engine_no_key):
        results = engine_no_key.search("test", source="news")
        for r in results:
            assert r.source_type == "news"

    def test_close_clears_client(self):
        engine = SearchEngine()
        engine.close()
        assert engine._client is None

    def test_server_source_not_news_labels_correctly(self, engine_no_key):
        """除 'news' 之外的 source 都标记为对应的 source_type。"""
        results = engine_no_key.search("test", source="web")
        assert all(r.source_type == "web" for r in results)


class TestSearchMockMethod:
    def test_formats_mock_results(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        engine = SearchEngine(api_key="")
        results = engine._search_mock("test query", source="web", max_results=3)
        assert len(results) == 3
        assert "模拟结果" in results[0].title
        assert "test query" in results[0].title
        assert results[0].url == "https://example.com/result-0"

    def test_mock_results_descending_relevance(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        engine = SearchEngine(api_key="")
        results = engine._search_mock("q", max_results=3)
        assert results[0].relevance_score > results[-1].relevance_score
