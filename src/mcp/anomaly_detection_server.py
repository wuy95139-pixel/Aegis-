"""
Anomaly Detection Server
========================
统计异常检测 — Z-score 和 IQR 两种方法，纯 Python 实现。

原理：
- Z-score: |x - mean| / std > threshold → 异常
- IQR: 值超出 [Q1 - 1.5*IQR, Q3 + 1.5*IQR] → 异常
- 双方法交集：同时被两种方法标记才认定为异常（减少误报）
"""

import logging
from typing import Dict, Any, List, Optional

from src.models.schemas import StructuredData
from src.mcp.data_reader import get_data_reader

logger = logging.getLogger(__name__)


class AnomalyDetectionServer:
    """异常检测服务 — 单例"""

    def _iter_numeric(
        self, data: StructuredData, column: str
    ) -> List[tuple]:
        """迭代数值列的值，保留原始行索引（跳过非数值行）。"""
        result = []
        for i, row in enumerate(data.rows):
            val = row.get(column, "")
            try:
                result.append((i, float(val)))
            except (ValueError, TypeError):
                continue
        return result

    def detect_zscore(
        self, data: StructuredData, column: str, threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Z-score 异常检测"""
        stats = data.column_stats.get(column)
        if not stats or stats.dtype != "numeric" or stats.std is None or stats.mean is None:
            return {"error": f"列 '{column}' 不是数值列或统计信息不足", "anomalies": [], "count": 0}

        indexed = self._iter_numeric(data, column)
        anomalies = []
        for orig_idx, v in indexed:
            if stats.std == 0:
                continue
            z = (v - stats.mean) / stats.std
            if abs(z) > threshold:
                anomalies.append({"row_index": orig_idx, "value": v, "z_score": round(z, 2)})

        return {
            "method": "zscore",
            "column": column,
            "threshold": threshold,
            "mean": round(stats.mean, 2),
            "std": round(stats.std, 2),
            "anomalies": anomalies,
            "count": len(anomalies),
            "anomaly_rate": round(len(anomalies) / max(len(indexed), 1) * 100, 1),
        }

    def detect_iqr(
        self, data: StructuredData, column: str, multiplier: float = 1.5
    ) -> Dict[str, Any]:
        """IQR 异常检测"""
        stats = data.column_stats.get(column)
        if not stats or stats.dtype != "numeric" or stats.q1 is None or stats.q3 is None:
            return {"error": f"列 '{column}' 不是数值列或统计信息不足", "anomalies": [], "count": 0}

        iqr = stats.q3 - stats.q1
        lower = stats.q1 - multiplier * iqr
        upper = stats.q3 + multiplier * iqr

        indexed = self._iter_numeric(data, column)

        anomalies = []
        for orig_idx, v in indexed:
            if v < lower or v > upper:
                anomalies.append({
                    "row_index": orig_idx, "value": v,
                    "bound": "lower" if v < lower else "upper",
                    "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
                })

        return {
            "method": "iqr",
            "column": column,
            "multiplier": multiplier,
            "q1": round(stats.q1, 2),
            "q3": round(stats.q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "anomalies": anomalies,
            "count": len(anomalies),
            "anomaly_rate": round(len(anomalies) / max(len(indexed), 1) * 100, 1),
        }

    def detect_all(
        self, data: StructuredData, method: str = "both",
        zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """对所有数值列运行异常检测"""
        reader = get_data_reader()
        numeric_cols = reader.get_numeric_columns(data)

        if not numeric_cols:
            return {"error": "没有检测到数值列", "columns": {}, "total_anomalies": 0}

        column_results = {}
        total_anomalies = 0
        anomaly_rows = set()

        for col in numeric_cols:
            z_result = self.detect_zscore(data, col, zscore_threshold)
            i_result = self.detect_iqr(data, col, iqr_multiplier)

            if "error" in z_result or "error" in i_result:
                column_results[col] = {"zscore": z_result, "iqr": i_result, "combined": []}
                continue

            # 双方法交集
            z_rows = {a["row_index"] for a in z_result.get("anomalies", [])}
            i_rows = {a["row_index"] for a in i_result.get("anomalies", [])}
            if method == "both":
                combined_rows = z_rows & i_rows
            elif method == "either":
                combined_rows = z_rows | i_rows
            else:
                combined_rows = set()

            anomaly_rows |= combined_rows
            combined = [{"row_index": ri, "row_data": data.rows[ri] if ri < len(data.rows) else {}}
                        for ri in sorted(combined_rows)]

            column_results[col] = {
                "zscore": {"count": z_result["count"], "anomaly_indices": sorted(z_rows)},
                "iqr": {"count": i_result["count"], "anomaly_indices": sorted(i_rows)},
                "combined": combined,
                "combined_count": len(combined),
            }
            total_anomalies += len(combined)

        return {
            "method": method,
            "columns": column_results,
            "total_anomalies": total_anomalies,
            "anomaly_row_indices": sorted(anomaly_rows),
            "numeric_columns_checked": numeric_cols,
        }

    def generate_summary(self, result: Dict[str, Any]) -> str:
        """生成中文异常检测摘要"""
        if result.get("error"):
            return f"异常检测失败: {result['error']}"

        parts = []
        parts.append(f"### 异常检测结果（{result['method']}方法）\n")

        for col, col_result in result.get("columns", {}).items():
            combined = col_result.get("combined", [])
            if combined:
                parts.append(
                    f"- **{col}**: {len(combined)} 个异常值 "
                    f"（Z-score: {col_result['zscore']['count']}个, "
                    f"IQR: {col_result['iqr']['count']}个, "
                    f"交集: {len(combined)}个）"
                )
                if len(combined) <= 10:
                    for a in combined:
                        rd = a.get("row_data", {})
                        parts.append(f"  - 行#{a['row_index']}: {rd.get(col, '?')}")
            else:
                parts.append(f"- **{col}**: 未发现显著异常")

        parts.append(f"\n> 共检测 {len(result.get('numeric_columns_checked', []))} 个数值列，"
                     f"发现 {result['total_anomalies']} 个异常数据点")
        return "\n".join(parts)


# 单例
_anomaly_instance: Optional[AnomalyDetectionServer] = None


def get_anomaly_server() -> AnomalyDetectionServer:
    global _anomaly_instance
    if _anomaly_instance is None:
        _anomaly_instance = AnomalyDetectionServer()
    return _anomaly_instance
