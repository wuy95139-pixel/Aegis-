"""
Analysis Planner
================
使用 LLM 制定分析步骤计划。

关键设计：LLM 只接收列名、类型、汇总统计（min/max/mean），
不接收原始数据值，确保 LLM 无法幻觉数据。
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from src.models.schemas import StructuredData

logger = logging.getLogger(__name__)


class AnalysisPlanner:
    """分析规划器 — 非单例，由 Orchestrator 传入 LLM"""

    def __init__(self, llm: Any):
        self.llm = llm

    def plan(self, data_list: List[StructuredData], user_intent: str = "") -> Dict[str, Any]:
        """制定分析步骤计划"""
        overview = self._build_overview(data_list, user_intent)

        prompt = f"""你是数据分析专家。根据数据概览制定分析计划。

{overview}

可用的分析操作：
- chart: 生成图表（bar/line/pie/scatter），需指定 x_column 和 y_column
- anomaly: 对数值列做统计异常检测（Z-score + IQR 双方法）
- what_if: What-If 场景模拟（对关键数值列做 ±百分比的变动分析）
- correlation: 分析数值列之间的相关性和关联模式
- summary: 生成数据统计汇总

请返回 JSON 格式的分析步骤列表。规划 6-12 步，覆盖上述多种分析类型。
每步格式：{{"action": "...", "description": "中文描述", "params": {{"chart_type": "...", "x_column": "...", "y_column": "..."}}}}

规则（务必遵守）：
- summary 步骤放第一步，总览数据
- 如果有日期/时间列，第二步用折线图展示核心数值列的趋势
- **每个低唯一值（≤30 个）的分类列至少对应一个 chart 步骤**：用该分类列做 x_column，分别搭配最重要的数值列（如销售额、利润）做柱状图或饼图。这是最重要的规则，不能遗漏任何分类列！
- 对于只有 2-3 个唯一值的分类列，优先用饼图展示占比
- 如果有"销售员"列，也需要做销售员业绩对比图
- 如果有"产品类别"列，除了销量对比，还应做利润/销售额对比
- anomaly 步骤对 1-2 个最重要的数值列做检测
- correlation 步骤对所有数值列做关联分析
- what_if 步骤对 1-2 个核心数值列模拟 ±10%/±20% 变化
- 优先分析用户意图中提到的列

只返回 JSON 数组，不要多余文字。"""

        resp = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        raw = resp.get("content", "[]").strip()
        # 提取 JSON
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        start = raw.find('[')
        end = raw.rfind(']')
        if start >= 0 and end > start:
            raw = raw[start:end + 1]

        try:
            steps = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Analysis planner JSON parse failed: {raw[:300]}")
            steps = self._default_plan(data_list)

        return {
            "overview": overview,
            "steps": steps,
            "step_count": len(steps),
        }

    def _build_overview(self, data_list: List[StructuredData], user_intent: str) -> str:
        """构建数据概览文本（元数据 + 统计摘要，不包含原始数据值）"""
        parts = []
        user_intent_line = f"\n用户意图: {user_intent}" if user_intent else ""
        parts.append(f"数据文件概览{user_intent_line}\n")

        for i, data in enumerate(data_list):
            sheet = f" (Sheet: {data.sheet_name})" if data.sheet_name else ""
            parts.append(f"## 数据集 {i + 1}: {data.filename}{sheet}")
            parts.append(f"- 行数: {data.row_count}, 列数: {data.col_count}")
            parts.append(f"- 列列表: {', '.join(data.columns)}")

            # 数值列统计
            parts.append(f"\n### 数值列统计:")
            for col in data.columns:
                stats = data.column_stats.get(col)
                if stats and stats.dtype == "numeric":
                    parts.append(
                        f"- **{col}**: min={stats.min}, max={stats.max}, "
                        f"mean={stats.mean}, median={stats.median}, std={stats.std}"
                    )

            # 字符串列概览
            string_cols = [c for c in data.columns
                          if data.column_stats.get(c) and data.column_stats[c].dtype != "numeric"]
            if string_cols:
                parts.append(f"\n### 分类/字符串列:")
                for col in string_cols:
                    stats = data.column_stats.get(col)
                    parts.append(
                        f"- **{col}**: 唯一值 {stats.unique_count} 个, "
                        f"示例: {', '.join(stats.sample_values[:3]) if stats else ''}"
                    )

            parts.append("")

        return "\n".join(parts)

    def _default_plan(self, data_list: List[StructuredData]) -> List[Dict[str, Any]]:
        """LLM 解析失败时的默认计划 — 覆盖所有分类列"""
        if not data_list:
            return []

        data = data_list[0]
        numeric_cols = [c for c in data.columns
                       if data.column_stats.get(c) and data.column_stats[c].dtype == "numeric"]
        string_cols = [c for c in data.columns if c not in numeric_cols]

        steps = [{"action": "summary", "description": "数据统计总览", "params": {}}]

        # 找出最重要的数值列（优先销售额、利润）
        priority_y = []
        for kw in ["销售额", "利润", "销量", "收入"]:
            for c in numeric_cols:
                if kw in c and c not in priority_y:
                    priority_y.append(c)
        # 补充其余数值列
        for c in numeric_cols:
            if c not in priority_y:
                priority_y.append(c)

        # 如果有日期列，加趋势图
        date_cols = [c for c in string_cols if any(kw in c for kw in ["日期", "时间", "date", "time"])]
        if date_cols and priority_y:
            steps.append({
                "action": "chart",
                "description": f"每日{priority_y[0]}趋势",
                "params": {"chart_type": "line", "x_column": date_cols[0], "y_column": priority_y[0]},
            })

        # 为每个非日期分类列（唯一值<=30）建图
        cat_cols = [c for c in string_cols
                    if c not in date_cols
                    and data.column_stats.get(c)
                    and data.column_stats[c].unique_count <= 30]

        for i, cat_col in enumerate(cat_cols):
            if priority_y:
                y_col = priority_y[min(i, len(priority_y) - 1)]
                uniq = data.column_stats[cat_col].unique_count if data.column_stats.get(cat_col) else 99
                ctype = "pie" if uniq <= 3 else "bar"
                steps.append({
                    "action": "chart",
                    "description": f"各{cat_col}{y_col}对比",
                    "params": {"chart_type": ctype, "x_column": cat_col, "y_column": y_col},
                })

        # 高级分析
        if len(numeric_cols) >= 1 and priority_y:
            steps.append({
                "action": "anomaly",
                "description": "检测异常数据点",
                "params": {"columns": priority_y[:2]},
            })
        if len(numeric_cols) >= 2:
            steps.append({
                "action": "correlation",
                "description": "分析数值列之间的相关性",
                "params": {"columns": numeric_cols},
            })
        if len(numeric_cols) >= 1 and priority_y:
            steps.append({
                "action": "what_if",
                "description": f"模拟 {priority_y[0]} 变化的场景",
                "params": {"column": priority_y[0], "change_pcts": [10, -10, 20, -20]},
            })

        return steps
