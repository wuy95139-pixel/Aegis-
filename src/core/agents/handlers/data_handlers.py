"""
数据处理与可视化处理器
=======================
从 Orchestrator 提取出的图表生成、看板创建、可视化分析处理器。

依赖 orchestrator_utils.py 中的静态工具函数。
"""

import json
import logging
from typing import Dict, Any, Optional

from src.core.agents.orchestrator_utils import (
    extract_json,
    validate_file_path,
    build_column_overview,
    build_smart_kpis,
    aggregate_chart_data,
)

logger = logging.getLogger(__name__)


class DataHandlers:
    """数据可视化相关所有意图处理器"""

    def __init__(self, llm):
        self.llm = llm

    # ===================== 图表生成 =====================

    def chart_generate(
        self, user_message: str, attached_file: Optional[str], params: dict
    ) -> dict:
        """生成图表（柱状图、折线图、饼图等）"""
        from src.mcp.quickchart_server import get_quickchart_server
        from src.mcp.data_reader import get_data_reader

        if not attached_file:
            return {"status": "error", "response": "请上传一个 CSV 或 Excel 数据文件，我会基于数据生成图表。"}

        reader = get_data_reader()
        try:
            all_data = reader.read_file(attached_file)
        except Exception as e:
            logger.warning(f"DataReader failed, fallback to LLM extraction: {e}")
            return self.chart_generate_fallback(user_message, attached_file, params)

        if not all_data:
            return {"status": "error", "response": "无法解析文件中的数据。"}

        data = all_data[0]
        numeric_cols = reader.get_numeric_columns(data)
        string_cols = reader.get_string_columns(data)

        if not numeric_cols:
            return {"status": "error", "response": "文件中没有检测到数值列，无法生成图表。"}

        chart_type = params.get("chart_type", "bar")

        overview = build_column_overview(data)
        prompt = f"""根据数据列概览，选择作图用的列。

数据概览：
{overview}

用户请求: {user_message}
建议图表类型: {chart_type}

可用数值列: {numeric_cols}
可用分类列: {string_cols}

返回 JSON:
{{"x_column": "X轴/标签列名", "y_column": "Y轴/数值列名", "chart_type": "{chart_type}", "title": "图表标题"}}

只返回 JSON。"""

        resp = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        raw = extract_json(resp["content"].strip())
        try:
            chart_cfg = json.loads(raw)
        except json.JSONDecodeError:
            chart_cfg = {}

        x_col = chart_cfg.get("x_column", string_cols[0] if string_cols else data.columns[0])
        y_col = chart_cfg.get("y_column", numeric_cols[0])
        ctype = chart_cfg.get("chart_type", chart_type)
        ctitle = chart_cfg.get("title", f"{y_col} by {x_col}")

        labels = [str(row.get(x_col, "")) for row in data.rows]
        values = [float(row.get(y_col, 0) or 0) for row in data.rows]

        qs = get_quickchart_server()
        if ctype in ("pie", "doughnut"):
            result = qs.pie_chart(labels, values, ctitle) if ctype == "pie" else qs.doughnut_chart(labels, values, ctitle)
        elif ctype == "line":
            result = qs.line_chart(labels, [{"label": y_col, "data": values}], ctitle)
        elif ctype == "radar":
            result = qs.radar_chart(labels, [{"label": y_col, "data": values}], ctitle)
        else:
            result = qs.bar_chart(labels, [{"label": y_col, "data": values}], ctitle)

        if not result.get("success"):
            return {"status": "error", "response": f"图表生成失败: {result.get('error', '未知错误')}"}

        img_tag = qs.chart_to_html_img(result, ctitle)
        return {
            "status": "success",
            "response": f"## 📊 {ctitle}\n\n{img_tag}\n\n> 数据来源: {data.filename} ({data.row_count} 行)",
        }

    def chart_generate_fallback(
        self, user_message: str, attached_file: str, params: dict
    ) -> dict:
        """旧版图表生成（回退方案：LLM 提取数据）"""
        from src.core.tools.file_tools import parse_file
        from src.mcp.quickchart_server import get_quickchart_server

        parsed = parse_file(attached_file)
        if not parsed.raw_text or parsed.raw_text.startswith("[不支持"):
            return {"status": "error", "response": "文件格式不支持，请上传 CSV 或 Excel 文件。"}

        file_data = parsed.raw_text
        chart_type = params.get("chart_type", "bar")
        title = params.get("title", "数据图表")

        prompt = f"""你是数据可视化专家。根据用户要求生成一个 Chart.js 图表配置（JSON 格式）。

用户请求: {user_message}
图表类型: {chart_type}
标题: {title}

可用数据：
{file_data[:4000]}

请返回 JSON 格式的图表配置，包含 type, title, labels, datasets。
注意：labels 和 data 值必须从上面数据中精确复制，不要修改数值。
只返回 JSON。"""

        resp = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = extract_json(resp["content"].strip())
        try:
            chart_cfg = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "error", "response": f"图表配置解析失败: {raw[:400]}"}

        qs = get_quickchart_server()
        labels = chart_cfg.get("labels", [])
        datasets = chart_cfg.get("datasets", [])
        ctype = chart_cfg.get("type", chart_type)
        ctitle = chart_cfg.get("title", title)

        if not labels or not datasets:
            return {"status": "error", "response": "数据不足，请提供包含数值的文件。"}

        if ctype in ("pie", "doughnut"):
            data_vals = datasets[0].get("data", []) if datasets else []
            result = qs.pie_chart(labels, data_vals, ctitle) if ctype == "pie" else qs.doughnut_chart(labels, data_vals, ctitle)
        elif ctype == "line":
            result = qs.line_chart(labels, datasets, ctitle)
        elif ctype == "radar":
            result = qs.radar_chart(labels, datasets, ctitle)
        else:
            result = qs.bar_chart(labels, datasets, ctitle)

        if not result.get("success"):
            return {"status": "error", "response": f"图表生成失败: {result.get('error', '未知错误')}"}

        img_tag = qs.chart_to_html_img(result, ctitle)
        return {
            "status": "success",
            "response": f"## 📊 {ctitle}\n\n{img_tag}\n\n> ⚠️ 使用 AI 辅助提取数据，结果可能有偏差",
        }

    # ===================== 数据看板 =====================

    def dashboard_create(
        self, user_message: str, attached_file: Optional[str], params: dict
    ) -> dict:
        """创建综合数据看板"""
        from src.mcp.dashboard_server import get_dashboard_server
        from src.mcp.quickchart_server import get_quickchart_server
        from src.mcp.data_reader import get_data_reader

        if not attached_file:
            return {
                "status": "error",
                "response": "请上传一个 CSV 或 Excel 数据文件，我会基于数据生成综合看板。",
            }

        reader = get_data_reader()
        try:
            all_data = reader.read_file(attached_file)
        except Exception as e:
            logger.warning(f"DataReader failed in dashboard: {e}")
            return {"status": "error", "response": f"文件解析失败: {e}"}

        if not all_data:
            return {"status": "error", "response": "无法解析文件中的数据。"}

        data = all_data[0]
        numeric_cols = reader.get_numeric_columns(data)
        string_cols = reader.get_string_columns(data)

        if not numeric_cols:
            return {"status": "error", "response": "文件中没有检测到数值列。"}

        title = params.get("title", f"{data.filename} 数据分析看板")

        overview = build_column_overview(data)
        prompt = f"""设计数据看板布局。

列概览：
{overview}

用户要求: {user_message}

可用数值列: {numeric_cols}
可用分类列: {string_cols}

返回 JSON:
{{
  "title": "看板标题",
  "subtitle": "副标题",
  "description": "2-3句分析要点概述",
  "kpi_columns": ["数值列名1", "数值列名2"],
  "charts": [{{"title": "图表标题", "type": "bar/line/pie", "x_column": "标签列", "y_column": "数值列"}}]
}}

KPI 用实际统计值，每张图表指定 x_column 和 y_column。
饼图适用于占比展示（一个 y_column），柱状图适用于对比。
设计 2-3 张互补图表。只返回 JSON。"""

        resp = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )

        raw = extract_json(resp["content"].strip())
        try:
            dash_cfg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Dashboard JSON parse failed, raw: {raw[:500]}")
            return {"status": "error", "response": f"看板配置解析失败: {raw[:400]}"}

        kpi_columns = dash_cfg.get("kpi_columns", numeric_cols[:4])
        kpis = []
        for col in kpi_columns:
            if col not in numeric_cols:
                continue
            stats = data.column_stats.get(col)
            if not stats:
                continue
            total = sum(float(row.get(col, 0) or 0) for row in data.rows)
            kpis.append({
                "label": f"总{col}",
                "value": f"{total:,.2f}" if total != int(total) else f"{total:,.0f}",
            })
            if stats.mean is not None:
                kpis.append({
                    "label": f"平均{col}",
                    "value": f"{stats.mean:,.2f}" if stats.mean != int(stats.mean) else f"{stats.mean:,.0f}",
                })

        ds = get_dashboard_server()
        charts_html_parts = []
        chart_configs_for_dashboard = []
        qs = get_quickchart_server()

        for i, chart_cfg in enumerate(dash_cfg.get("charts", [])):
            ctype = chart_cfg.get("type", "bar")
            x_col = chart_cfg.get("x_column", string_cols[0] if string_cols else data.columns[0])
            y_col = chart_cfg.get("y_column", numeric_cols[0])
            ctitle = chart_cfg.get("title", f"图表 {i + 1}")

            clabels = [str(row.get(x_col, "")) for row in data.rows]
            cvalues = [float(row.get(y_col, 0) or 0) for row in data.rows]

            if ctype in ("pie", "doughnut"):
                cr = qs.pie_chart(clabels, cvalues, ctitle) if ctype == "pie" else qs.doughnut_chart(clabels, cvalues, ctitle)
            elif ctype == "line":
                cr = qs.line_chart(clabels, [{"label": y_col, "data": cvalues}], ctitle)
            else:
                cr = qs.bar_chart(clabels, [{"label": y_col, "data": cvalues}], ctitle)

            chart_configs_for_dashboard.append({
                "title": ctitle, "type": ctype, "labels": clabels,
                "datasets": [{"label": y_col, "data": cvalues}],
            })

            if cr.get("success"):
                charts_html_parts.append(f"### {ctitle}\n\n{qs.chart_to_html_img(cr, ctitle)}\n")

        result = ds.create_dashboard(
            title=dash_cfg.get("title", title),
            subtitle=dash_cfg.get("subtitle", ""),
            charts=chart_configs_for_dashboard,
            kpis=kpis,
            description=dash_cfg.get("description", ""),
        )

        kpi_table_lines = []
        if kpis:
            kpi_table_lines = ["| 指标 | 数值 |", "|------|------|"]
            for kpi in kpis:
                kpi_table_lines.append(f"| {kpi['label']} | {kpi['value']} |")

        kpi_text = "### 📈 关键指标\n\n" + "\n".join(kpi_table_lines) + "\n" if kpi_table_lines else ""
        charts_html = "\n".join(charts_html_parts)
        url_path = result.get("url_path", "")

        response_text = (
            f"## 📊 {dash_cfg.get('title', title)}\n\n"
            f"> {dash_cfg.get('subtitle', '')}\n\n"
            f"{kpi_text}\n"
            f"{charts_html}\n"
            f"### 📋 分析摘要\n\n{dash_cfg.get('description', '')}\n\n"
            f"---\n"
            f"[在新窗口打开完整看板]({url_path})"
        )

        return {"status": "success", "response": response_text}

    # ===================== 综合可视化分析 =====================

    def visual_analysis(
        self, user_message: str, attached_file: Optional[str], params: dict
    ) -> dict:
        """综合可视化分析 — 多步骤数据探索流程"""
        from src.mcp.data_reader import get_data_reader
        from src.mcp.analysis_planner import AnalysisPlanner
        from src.mcp.quickchart_server import get_quickchart_server
        from src.mcp.anomaly_detection_server import get_anomaly_server
        from src.mcp.recommendation_server import get_recommendation_server
        from src.mcp.what_if_server import get_what_if_server
        from src.mcp.report_builder import get_report_builder

        if not attached_file:
            return {"status": "error", "response": "请上传一个 CSV 或 Excel 数据文件进行可视化分析。"}

        validate_file_path(attached_file)
        reader = get_data_reader()
        try:
            all_data = reader.read_file(attached_file)
        except Exception as e:
            return {"status": "error", "response": f"文件解析失败: {e}"}

        if not all_data:
            return {"status": "error", "response": "无法解析文件中的数据。"}

        data = all_data[0]
        numeric_cols = reader.get_numeric_columns(data)
        string_cols = reader.get_string_columns(data)

        if not numeric_cols:
            return {"status": "error", "response": "文件中没有检测到数值列，无法进行可视化分析。"}

        logger.info(f"Visual analysis: {data.filename} - {data.row_count} rows, "
                    f"{len(numeric_cols)} numeric cols, {len(string_cols)} string cols")

        response_parts = [
            f"## 📊 综合分析: {data.filename}\n",
            f"**数据维度**: {data.row_count} 行 × {data.col_count} 列\n",
        ]
        if data.sheet_name:
            response_parts.append(f"**Sheet**: {data.sheet_name}\n")

        col_lines = ["### 列概览\n"]
        for col in data.columns:
            stats = data.column_stats.get(col)
            if stats and stats.dtype == "numeric":
                col_lines.append(f"- **{col}** (数值): min={stats.min}, max={stats.max}, 均值={stats.mean}")
            else:
                uv = stats.unique_count if stats else "?"
                col_lines.append(f"- **{col}** (分类): {uv} 个唯一值")
        response_parts.append("\n".join(col_lines) + "\n")

        planner = AnalysisPlanner(self.llm)
        plan_result = planner.plan(all_data, user_message)
        steps = plan_result.get("steps", [])

        response_parts.append(f"### 📋 分析计划（共 {len(steps)} 步）\n")
        for i, step in enumerate(steps):
            response_parts.append(f"{i + 1}. {step.get('description', step.get('action', '?'))}\n")
        response_parts.append("---\n")

        qs = get_quickchart_server()
        report_builder = get_report_builder()
        chart_cards = []
        anomaly_result = None
        corr_result = None
        whatif_results = []

        for step in steps:
            action = step.get("action", "")
            desc = step.get("description", "")
            sp = step.get("params", {})

            if action == "chart":
                x_col = sp.get("x_column", string_cols[0] if string_cols else data.columns[0])
                y_col = sp.get("y_column", numeric_cols[0])
                ctype = sp.get("chart_type", "bar")
                ctitle = desc or f"{y_col} by {x_col}"

                clabels, cvalues = aggregate_chart_data(data, x_col, y_col, ctype)

                if ctype in ("pie", "doughnut"):
                    cr = qs.pie_chart(clabels, cvalues, ctitle) if ctype == "pie" else qs.doughnut_chart(clabels, cvalues, ctitle)
                elif ctype == "line":
                    cr = qs.line_chart(clabels, [{"label": y_col, "data": cvalues}], ctitle)
                else:
                    cr = qs.bar_chart(clabels, [{"label": y_col, "data": cvalues}], ctitle)

                if cr.get("success"):
                    img_html = qs.chart_to_html_img(cr, ctitle)
                    response_parts.append(f"### {ctitle}\n\n{img_html}\n")
                    chart_cards.append(report_builder.build_chart_card(ctitle, img_html))

            elif action == "anomaly":
                asrv = get_anomaly_server()
                anomaly_result = asrv.detect_all(data, method="both")
                summary = asrv.generate_summary(anomaly_result)
                response_parts.append(summary + "\n")

            elif action == "correlation":
                rsrv = get_recommendation_server()
                corr_result = rsrv.compute_correlations(data)
                insights = rsrv.generate_insights({"correlations": corr_result})
                if insights:
                    response_parts.append("### 🔗 关联分析\n")
                    for ins in insights:
                        response_parts.append(f"- {ins}\n")
                    response_parts.append("")

            elif action == "what_if":
                wsrv = get_what_if_server()
                col = sp.get("column", numeric_cols[0])
                pcts = sp.get("change_pcts", [10, -10])
                for pct in pcts:
                    sim = wsrv.simulate_percentage_change(data, col, pct)
                    sim["label"] = f"{col}变化{pct:+.0f}%"
                    summary = wsrv.generate_summary(sim)
                    response_parts.append(summary + "\n")
                    whatif_results.append(sim)

            elif action == "summary":
                stats_lines = ["### 📈 数据统计\n"]
                stats_lines.append("| 列名 | 类型 | 最小值 | 最大值 | 均值 | 中位数 |")
                stats_lines.append("|------|------|--------|--------|------|--------|")
                for col in numeric_cols:
                    s = data.column_stats.get(col)
                    if s:
                        stats_lines.append(f"| {col} | 数值 | {s.min} | {s.max} | {s.mean} | {s.median} |")
                response_parts.append("\n".join(stats_lines) + "\n")

        kpis = build_smart_kpis(data, numeric_cols)

        findings_parts = []
        for kpi in kpis:
            findings_parts.append(f"- {kpi['label']}: {kpi['value']}")

        if corr_result and corr_result.get("pairs"):
            strong = [p for p in corr_result["pairs"] if abs(p["correlation"]) >= 0.4]
            if strong:
                findings_parts.append("\n关联发现:")
                for p in strong[:5]:
                    direction = "正相关" if p["correlation"] > 0 else "负相关"
                    findings_parts.append(f"- {p['col1']} 与 {p['col2']}: {direction} (r={p['correlation']})")

        if anomaly_result:
            total_anom = anomaly_result.get("total_anomalies", 0)
            findings_parts.append(f"\n异常检测: 共发现 {total_anom} 个异常数据点")

        for w in whatif_results[:3]:
            label = w.get("label", "")
            change = w.get("change", {})
            if change.get("sum_delta") is not None:
                findings_parts.append(f"- {label}: 总和变化 {change.get('sum_delta_pct', '?')}%")

        findings_str = "\n".join(findings_parts)

        cat_info = []
        for col in string_cols[:5]:
            stats = data.column_stats.get(col)
            if stats:
                cat_info.append(f"- {col}: {stats.unique_count} 个类别, 示例: {', '.join(str(x) for x in stats.sample_values[:3])}")

        narrative_prompt = f"""你是数据分析师。基于以下真实计算结果，写一段 5-8 句话的中文分析总结。

数据文件: {data.filename}
数据规模: {data.row_count} 行 × {data.col_count} 列
分类列: {', '.join(string_cols[:5])}
数值列: {', '.join(numeric_cols[:5])}

==== 真实计算结果 ====
{findings_str}

分类列概况:
{chr(10).join(cat_info)}

==== 请据此撰写分析总结 ====
要求：
1. 引用具体的数值（如总销售额、总利润等），这些数字来自真实计算
2. 如果有关联分析发现，解释强关联的业务含义
3. 如果有异常数据点，说明可能的影响
4. 保持专业简洁，每句话有信息量
5. 用中文输出，纯文本，不要 markdown 格式"""

        narrative_resp = self.llm.chat(
            messages=[{"role": "user", "content": narrative_prompt}],
            temperature=0.4,
            max_tokens=600,
        )
        narrative = narrative_resp.get("content", "").strip()
        if narrative:
            response_parts.append(f"### 💡 分析总结\n\n{narrative}\n")

        anomaly_html = report_builder.build_anomaly_section(anomaly_result) if anomaly_result else ""
        corr_html = report_builder.build_correlation_section(corr_result) if corr_result else ""
        whatif_html = report_builder.build_whatif_section(whatif_results)
        kpi_html = report_builder.build_kpi_grid(kpis)
        data_overview_html = (
            f"<p>{data.filename} · {data.row_count} 行 × {data.col_count} 列 · "
            f"{len(numeric_cols)} 个数值列 · {len(string_cols)} 个分类列</p>"
        )

        title = params.get("title", f"{data.filename} 综合分析报告")
        meta_info = (
            f"<span>📋 {data.row_count} 条记录</span>"
            f"<span>📊 {data.col_count} 列</span>"
            f"<span>🔢 {len(numeric_cols)} 个数值列</span>"
        )
        report = report_builder.build_report(
            title=title,
            subtitle=f"自动生成 · {len(steps)} 步分析",
            kpi_html=kpi_html,
            chart_sections=chart_cards,
            anomaly_html=anomaly_html,
            correlation_html=corr_html,
            whatif_html=whatif_html,
            narrative=narrative,
            data_overview=data_overview_html,
            meta_info=meta_info,
        )

        if report.get("success"):
            response_parts.append(f"---\n📁 [完整 HTML 报告]({report['url_path']})\n")

        return {"status": "success", "response": "\n".join(response_parts)}
