"""
Orchestrator 工具函数
====================
从 Orchestrator 提取出的纯函数工具，无状态、无副作用。
"""

import os as _os
import re
from typing import Optional, Any


# ===================== JSON 提取 =====================

def extract_json(raw: str) -> str:
    """从 LLM 响应中健壮地提取 JSON 字符串。

    处理常见情况：
    1. ```json ... ``` 或 ``` ... ``` 代码块包裹
    2. 代码块前后有额外文字
    3. JSON 嵌入在其他文字中 (提取 { 到 } 的最外层)
    """
    raw = raw.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    return raw


# ===================== 路径验证 =====================

def validate_file_path(filepath: str) -> str:
    """防止路径遍历：确保文件路径在允许的目录内。"""
    data_dir = _os.environ.get("AEGIS_DATA_DIR", "./data")
    allowed_dirs = [
        _os.path.realpath(_os.path.join(data_dir, "uploads")),
        _os.path.realpath(_os.path.join(data_dir, "reports")),
        _os.path.realpath(_os.path.join(data_dir, "dashboards")),
        _os.path.realpath("./output"),
    ]
    real_path = _os.path.realpath(filepath)
    for allowed in allowed_dirs:
        if real_path.startswith(allowed + _os.sep) or real_path == allowed:
            return real_path
    raise ValueError(f"文件路径超出允许范围: {filepath}")


# ===================== 内容提取 =====================

def get_content(
    filepath: Optional[str],
    msg: str,
    extracted_text: Optional[str] = None,
) -> str:
    """获取要处理的文本内容：LLM提取优先 > 文件 > 冒号分割"""
    if filepath:
        from src.core.tools.file_tools import parse_file
        parsed = parse_file(filepath)
        return parsed.raw_text
    if extracted_text and len(extracted_text) > 1:
        return extracted_text
    if ":" in msg or "：" in msg:
        parts = msg.split(":", 1) if ":" in msg else msg.split("：", 1)
        if len(parts) > 1 and len(parts[1].strip()) > 1:
            return parts[1].strip()
    return msg


# ===================== 数据可视化静态工具 =====================

def build_smart_kpis(data: Any, numeric_cols: list) -> list:
    """智能选择业务关键 KPI：优先销售额/利润/销量，计算毛利率等派生指标"""
    kpis = []
    col_set = {col.lower(): col for col in numeric_cols}

    for kw in ["销售额", "收入", "营收", "金额", "总价"]:
        if kw in col_set:
            col = col_set[kw]
            total = sum(float(row.get(col, 0) or 0) for row in data.rows)
            kpis.append({"label": f"总{col}", "value": f"¥{total:,.0f}"})
            break

    for kw in ["利润", "毛利", "净利"]:
        if kw in col_set:
            col = col_set[kw]
            total = sum(float(row.get(col, 0) or 0) for row in data.rows)
            kpis.append({"label": f"总{col}", "value": f"¥{total:,.0f}"})
            break

    for kw in ["销量", "数量", "件数", "销售量"]:
        if kw in col_set:
            col = col_set[kw]
            total = sum(float(row.get(col, 0) or 0) for row in data.rows)
            kpis.append({"label": f"总{col}", "value": f"{total:,.0f}"})
            break

    profit_col = sales_col = None
    for kw in ["利润", "毛利"]:
        if kw in col_set:
            profit_col = col_set[kw]
            break
    for kw in ["销售额", "收入", "营收"]:
        if kw in col_set:
            sales_col = col_set[kw]
            break
    if profit_col and sales_col:
        total_profit = sum(float(row.get(profit_col, 0) or 0) for row in data.rows)
        total_sales = sum(float(row.get(sales_col, 0) or 0) for row in data.rows)
        if total_sales > 0:
            margin = (total_profit / total_sales) * 100
            kpis.append({"label": "毛利率", "value": f"{margin:.1f}%"})

    return kpis[:4]


def aggregate_chart_data(data: Any, x_col: str, y_col: str, chart_type: str):
    """
    为图表准备数据。如果 x 轴是分类列，按类别聚合 y 值；
    如果是日期/数值列，保留明细用于趋势图。
    返回 (labels, values) 元组。
    """
    x_stats = data.column_stats.get(x_col) if data.column_stats else None
    is_categorical = x_stats and x_stats.dtype != "numeric"

    if is_categorical and chart_type in ("bar", "pie", "doughnut"):
        groups: dict = {}
        for row in data.rows:
            key = str(row.get(x_col, "")).strip()
            if not key:
                continue
            try:
                val = float(row.get(y_col, 0) or 0)
            except (ValueError, TypeError):
                continue
            groups[key] = groups.get(key, 0) + val
        labels = list(groups.keys())
        values = [groups[k] for k in labels]
    elif is_categorical and chart_type == "line":
        labels = [str(row.get(x_col, "")) for row in data.rows]
        values = [float(row.get(y_col, 0) or 0) for row in data.rows]
    else:
        labels = [str(row.get(x_col, "")) for row in data.rows]
        values = [float(row.get(y_col, 0) or 0) for row in data.rows]

    return labels, values


def build_column_overview(data: Any) -> str:
    """构建列概览文本（用于 LLM 列选择）"""
    lines = []
    for col in data.columns:
        stats = data.column_stats.get(col)
        if stats and stats.dtype == "numeric":
            lines.append(
                f"- {col} (数值): min={stats.min}, max={stats.max}, mean={stats.mean}"
            )
        else:
            samples = stats.sample_values[:3] if stats else []
            lines.append(
                f"- {col} (分类): 唯一值{stats.unique_count if stats else '?'}个, 示例: {samples}"
            )
    return "\n".join(lines)
