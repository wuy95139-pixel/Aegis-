"""
Recommendation / Association Server
===================================
关联分析 — 数值列之间的 Pearson 相关性，分类列的共现模式。

不依赖 scikit-learn，使用纯 Python 和 statistics 模块实现。
"""

import logging
from typing import Dict, Any, List, Optional
from collections import Counter

from src.models.schemas import StructuredData
from src.mcp.data_reader import get_data_reader
from src.utils.common import pearson_correlation

logger = logging.getLogger(__name__)


class RecommendationServer:
    """关联分析服务 — 单例"""

    def compute_correlations(self, data: StructuredData) -> Dict[str, Any]:
        """计算所有数值列两两之间的 Pearson 相关系数"""
        reader = get_data_reader()
        numeric_cols = reader.get_numeric_columns(data)

        if len(numeric_cols) < 2:
            return {"error": "至少需要 2 个数值列才能计算相关性", "pairs": [], "matrix": {}}

        # 按行配对提取数值：只计算两列同时有有效数值的行
        # 避免 get_column_values 跳过不同行导致的值错位
        def _paired_values(col1: str, col2: str) -> tuple:
            xs, ys = [], []
            for row in data.rows:
                try:
                    xs.append(float(row.get(col1, "")))
                    ys.append(float(row.get(col2, "")))
                except (ValueError, TypeError):
                    continue
            return xs, ys

        pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                xs, ys = _paired_values(col1, col2)
                if len(xs) < 3:
                    continue
                corr = pearson_correlation(xs, ys)
                if corr is not None:
                    pairs.append({
                        "col1": col1, "col2": col2,
                        "correlation": round(corr, 3),
                        "strength": self._label_strength(corr),
                    })

        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

        return {
            "numeric_columns": numeric_cols,
            "pairs": pairs,
            "pair_count": len(pairs),
        }

    def _label_strength(self, corr: float) -> str:
        """标注相关性强弱"""
        r = abs(corr)
        if r >= 0.7:
            return "strong_positive" if corr > 0 else "strong_negative"
        elif r >= 0.4:
            return "moderate_positive" if corr > 0 else "moderate_negative"
        return "weak"

    def find_co_occurrence(
        self, data: StructuredData, categorical_column: str, top_n: int = 10
    ) -> Dict[str, Any]:
        """
        分析分类列中各值与其他列的共现模式。
        对于 categorical_column 的每个唯一值，统计其他列中出现频率最高的值。
        """
        if categorical_column not in data.columns:
            return {"error": f"列 '{categorical_column}' 不存在"}

        reader = get_data_reader()
        other_cols = [c for c in data.columns if c != categorical_column]

        # 按分类列的值分组
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in data.rows:
            key = str(row.get(categorical_column, "")).strip()
            if key:
                groups.setdefault(key, []).append(row)

        results = []
        for key, group_rows in groups.items():
            key_info = {"value": key, "count": len(group_rows),
                         "associations": [], "numeric_averages": {}}

            for col in other_cols:
                stats = data.column_stats.get(col)
                if stats and stats.dtype == "numeric":
                    vals = []
                    for r in group_rows:
                        raw = r.get(col)
                        if raw is None or str(raw).strip() == "":
                            continue
                        try:
                            vals.append(float(raw))
                        except (ValueError, TypeError):
                            pass
                    if vals:
                        key_info["numeric_averages"][col] = round(sum(vals) / len(vals), 2)
                else:
                    vals = [str(r.get(col, "")).strip() for r in group_rows]
                    counter = Counter(v for v in vals if v)
                    key_info["associations"].append({
                        "column": col,
                        "top_values": [{"value": v, "count": c}
                                       for v, c in counter.most_common(3)],
                    })

            results.append(key_info)

        return {
            "column": categorical_column,
            "unique_values": len(groups),
            "groups": results,
        }

    def analyze_all(self, data: StructuredData) -> Dict[str, Any]:
        """综合运行所有关联分析"""
        corr_result = self.compute_correlations(data)

        # 对每个低基数（<=20 个唯一值）的字符串列做共现分析
        reader = get_data_reader()
        string_cols = reader.get_string_columns(data)
        cooccurrence = {}
        for col in string_cols:
            stats = data.column_stats.get(col)
            if stats and stats.unique_count <= 20:
                cooccurrence[col] = self.find_co_occurrence(data, col)

        return {
            "correlations": corr_result,
            "co_occurrence": cooccurrence,
        }

    def generate_insights(self, result: Dict[str, Any]) -> List[str]:
        """从关联分析结果生成中文洞察列表"""
        insights = []
        pairs = result.get("correlations", {}).get("pairs", [])

        for p in pairs:
            corr = p["correlation"]
            if abs(corr) >= 0.7:
                direction = "正相关" if corr > 0 else "负相关"
                insights.append(
                    f"**{p['col1']}** 与 **{p['col2']}** 存在强{direction} "
                    f"(r={corr}) — "
                    f"{'一个增大另一个也增大' if corr > 0 else '一个增大另一个减小'}"
                )
            elif abs(corr) >= 0.4:
                direction = "正相关" if corr > 0 else "负相关"
                insights.append(
                    f"**{p['col1']}** 与 **{p['col2']}** 存在中等{direction} "
                    f"(r={corr})"
                )

        if not insights:
            insights.append("未发现显著相关性（|r| >= 0.4）")

        return insights


# 单例
_recommendation_instance: Optional[RecommendationServer] = None


def get_recommendation_server() -> RecommendationServer:
    global _recommendation_instance
    if _recommendation_instance is None:
        _recommendation_instance = RecommendationServer()
    return _recommendation_instance
