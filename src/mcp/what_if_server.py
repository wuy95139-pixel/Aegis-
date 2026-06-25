"""
What-If Simulation Server
=========================
场景模拟 — 对数值列施加百分比变化或阈值替换，估算对其他列的影响。

影响估算：使用 Pearson 相关系数和标准差比率做线性近似。
ΔY ≈ corr * (σY / σX) * ΔX  （基于简单线性回归斜率）
"""

import logging
import math
from typing import Dict, Any, List, Optional

from src.models.schemas import StructuredData
from src.mcp.data_reader import get_data_reader
from src.utils.common import pearson_correlation

logger = logging.getLogger(__name__)


class WhatIfServer:
    """What-If 模拟服务 — 单例"""

    def simulate_percentage_change(
        self, data: StructuredData, target_column: str, change_pct: float,
        impact_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        对 target_column 施加 change_pct% 的变化，估算影响。
        正值=增加，负值=减少。例如 change_pct=10 表示所有值 +10%。
        """
        reader = get_data_reader()
        numeric_cols = reader.get_numeric_columns(data)

        if target_column not in numeric_cols:
            return {"error": f"'{target_column}' 不是数值列"}
        if target_column not in data.column_stats:
            return {"error": f"'{target_column}' 统计信息不可用"}

        stats = data.column_stats[target_column]
        values = reader.get_column_values(data, target_column, typed=True)
        factor = 1 + change_pct / 100.0

        # 目标列变化后的值
        new_values = [v * factor for v in values]
        new_total = sum(new_values)
        old_total = sum(values)

        result = {
            "scenario": f"{target_column} 变化 {change_pct:+.1f}%",
            "target_column": target_column,
            "change_pct": change_pct,
            "original": {
                "sum": round(old_total, 2),
                "mean": round(stats.mean, 2) if stats.mean is not None else None,
                "min": round(stats.min, 2) if stats.min is not None else None,
                "max": round(stats.max, 2) if stats.max is not None else None,
            },
            "new": {
                "sum": round(new_total, 2),
                "mean": round(sum(new_values) / len(new_values), 2) if new_values else None,
                "min": round(min(new_values), 2),
                "max": round(max(new_values), 2),
            },
            "change": {
                "sum_delta": round(new_total - old_total, 2),
                "sum_delta_pct": round((new_total - old_total) / max(abs(old_total), 1) * 100, 1),
            },
            "impacts": {},
        }

        # 估算对其他列的影响
        if impact_columns is None:
            impact_columns = [c for c in numeric_cols if c != target_column]

        for col in impact_columns:
            impact = self._estimate_impact(
                data, target_column, col, change_pct, stats.std
            )
            if impact:
                result["impacts"][col] = impact

        return result

    def _estimate_impact(
        self, data: StructuredData, from_col: str, to_col: str,
        change_pct: float, from_std: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """使用 Pearson 相关系数估算变化对另一列的影响"""
        reader = get_data_reader()
        from_vals = reader.get_column_values(data, from_col, typed=True)
        to_vals = reader.get_column_values(data, to_col, typed=True)

        if len(from_vals) < 3 or len(to_vals) < 3:
            return None

        to_stats = data.column_stats.get(to_col)
        if not to_stats or to_stats.mean is None or to_stats.std is None:
            return None

        corr = pearson_correlation(from_vals, to_vals)
        if corr is None:
            return None

        old_mean = to_stats.mean
        from_mean = data.column_stats[from_col].mean
        if from_std and from_std > 0 and from_mean is not None and from_mean != 0:
            # ΔY ≈ corr * σY * (Δ% * μX / σX) — 标准线性回归斜率近似
            delta_pct = change_pct / 100.0
            relative_change = corr * to_stats.std * delta_pct * (from_mean / from_std)
        else:
            # 无法计算缩放因子时，用弹性近似: ΔY% ≈ corr * ΔX%
            delta_pct = change_pct / 100.0
            relative_change = corr * old_mean * delta_pct

        new_mean = old_mean + relative_change

        return {
            "correlation": round(corr, 3),
            "original_mean": round(old_mean, 2),
            "estimated_new_mean": round(new_mean, 2),
            "delta": round(relative_change, 2),
            "delta_pct": round(relative_change / max(abs(old_mean), 1) * 100, 1),
            "note": "基于线性相关性估算" if abs(corr) < 0.99 else None,
        }

    def run_scenarios(
        self, data: StructuredData, scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """批量运行多个场景"""
        results = []
        for i, sc in enumerate(scenarios):
            col = sc.get("column")
            pct = sc.get("change_pct", 0)
            impacts = sc.get("impact_columns")
            label = sc.get("label", f"场景{i + 1}")

            if not col:
                results.append({"label": label, "error": "缺少 column 参数"})
                continue

            sim = self.simulate_percentage_change(data, col, pct, impacts)
            sim["label"] = label
            results.append(sim)

        return {"scenarios": results, "count": len(results)}

    def generate_summary(self, result: Dict[str, Any]) -> str:
        """生成中文 What-If 摘要"""
        if result.get("error"):
            return f"What-If 模拟失败: {result['error']}"

        parts = []
        label = result.get("label", result.get("scenario", "模拟"))
        parts.append(f"#### {label}")

        orig = result.get("original", {})
        new = result.get("new", {})
        change = result.get("change", {})
        parts.append(
            f"| 指标 | 原始 | 变化后 | 变动 |\n"
            f"|------|------|--------|------|\n"
            f"| 总和 | {orig.get('sum', '-')} | {new.get('sum', '-')} | {change.get('sum_delta', '-')} "
            f"({change.get('sum_delta_pct', '-')}%) |\n"
            f"| 均值 | {orig.get('mean', '-')} | {new.get('mean', '-')} | - |\n"
            f"| 最小值 | {orig.get('min', '-')} | {new.get('min', '-')} | - |\n"
            f"| 最大值 | {orig.get('max', '-')} | {new.get('max', '-')} | - |"
        )

        if result.get("impacts"):
            parts.append(f"\n**对其他列的影响估算**:")
            for col, impact in result["impacts"].items():
                note = f" ({impact['note']})" if impact.get("note") else ""
                parts.append(
                    f"- {col}: 预估均值 {impact['original_mean']} → "
                    f"{impact['estimated_new_mean']} "
                    f"({impact['delta_pct']:+.1f}%)，相关系数 r={impact['correlation']}{note}"
                )

        return "\n".join(parts)


# 单例
_what_if_instance: Optional[WhatIfServer] = None


def get_what_if_server() -> WhatIfServer:
    global _what_if_instance
    if _what_if_instance is None:
        _what_if_instance = WhatIfServer()
    return _what_if_instance
