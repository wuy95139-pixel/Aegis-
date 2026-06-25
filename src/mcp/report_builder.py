"""
Report Builder
==============
生成综合分析 HTML 报告，Apple-style 简洁设计。
"""

import html as _html
import logging
import re
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def _esc(s: Any) -> str:
    """HTML-escape user data to prevent XSS injection."""
    return _html.escape(str(s), quote=True)

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
    line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; }}

  /* Header */
  .header {{
    background: linear-gradient(135deg, #1d1d1f 0%, #2d2d2f 100%);
    color: white;
    padding: 72px 0 56px;
    text-align: center;
  }}
  .header h1 {{
    font-size: 44px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 12px;
  }}
  .header .subtitle {{
    font-size: 19px;
    color: #a1a1a6;
    font-weight: 300;
  }}
  .header .meta {{
    margin-top: 18px;
    font-size: 14px;
    color: #6e6e73;
  }}
  .header .meta span {{ margin: 0 12px; }}

  /* Stats Bar */
  .stats-bar {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin: -40px auto 40px;
    padding: 0 24px;
    max-width: 1200px;
  }}
  .stat-card {{
    background: white;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }}
  .stat-card .icon {{ font-size: 28px; margin-bottom: 8px; }}
  .stat-card .value {{
    font-size: 28px;
    font-weight: 700;
    color: #1d1d1f;
    word-break: break-all;
  }}
  .stat-card .label {{
    font-size: 13px;
    color: #6e6e73;
    margin-top: 4px;
  }}
  .stat-card.highlight {{
    background: linear-gradient(135deg, #007AFF, #5856D6);
  }}
  .stat-card.highlight .value,
  .stat-card.highlight .label,
  .stat-card.highlight .icon {{ color: white; }}

  /* Section */
  .section {{ margin: 48px 0; }}
  .section-title {{
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: -0.3px;
  }}
  .section-desc {{
    color: #6e6e73;
    font-size: 16px;
    margin-bottom: 28px;
  }}

  /* Card Grid */
  .card-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  .card {{
    background: white;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h3 {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    color: #1d1d1f;
  }}
  .card img {{
    width: 100%;
    border-radius: 12px;
    max-width: 480px;
  }}

  /* Insight List */
  .insight-list {{ list-style: none; }}
  .insight-list li {{
    padding: 14px 0;
    border-bottom: 1px solid #f0f0f2;
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }}
  .insight-list li:last-child {{ border-bottom: none; }}
  .insight-list .bullet {{
    width: 8px; height: 8px;
    background: #007AFF;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 8px;
  }}
  .insight-list .label {{
    font-weight: 600;
    color: #1d1d1f;
  }}
  .insight-list .detail {{
    color: #6e6e73;
  }}

  /* Data Table */
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}
  .data-table th {{
    background: #f5f5f7;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    color: #1d1d1f;
    border-bottom: 2px solid #e0e0e0;
  }}
  .data-table td {{
    padding: 12px 16px;
    border-bottom: 1px solid #f0f0f2;
  }}
  .data-table tr:hover {{ background: #fafafa; }}

  /* Highlight Box */
  .highlight-box {{
    background: linear-gradient(135deg, #EBF5FF, #F0EBFF);
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
  }}
  .highlight-box h4 {{
    color: #007AFF;
    font-size: 16px;
    margin-bottom: 8px;
  }}

  /* Advanced Analytics Cards */
  .adv-card {{
    background: white;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 20px;
  }}
  .adv-card h3 {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #1d1d1f;
  }}
  .adv-card .badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
  }}
  .badge-blue {{ background: #EBF5FF; color: #007AFF; }}
  .badge-green {{ background: #E8F8E8; color: #34C759; }}
  .badge-orange {{ background: #FFF3E0; color: #FF9500; }}
  .badge-red {{ background: #FFE5E5; color: #FF3B30; }}

  /* Summary bullets with colors */
  .summary-bullets {{ list-style: none; }}
  .summary-bullets li {{
    padding: 14px 0;
    border-bottom: 1px solid #f0f0f2;
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }}
  .summary-bullets li:last-child {{ border-bottom: none; }}
  .summary-bullets .s-bullet {{
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 7px;
  }}
  .summary-bullets .label {{
    font-weight: 600;
    color: #1d1d1f;
  }}
  .summary-bullets .detail {{
    color: #6e6e73;
  }}

  /* Footer */
  .footer {{
    text-align: center;
    padding: 40px 0;
    color: #6e6e73;
    font-size: 13px;
    border-top: 1px solid #e0e0e0;
    margin-top: 60px;
  }}

  @media (max-width: 768px) {{
    .header h1 {{ font-size: 32px; }}
    .stats-bar {{ grid-template-columns: repeat(2, 1fr); }}
    .card-grid {{ grid-template-columns: 1fr; }}
    .stat-card .value {{ font-size: 22px; }}
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="container">
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    {meta_html}
  </div>
</div>

<div class="container">

{stats_bar}

{body_content}

</div>

<!-- Footer -->
<div class="footer">
  <div class="container">
    <p>Aegis Analysis Report &middot; 数据驱动决策</p>
    <p>生成于 {generated_at} &middot; 数据仅供内部参考</p>
  </div>
</div>
</body>
</html>"""


class ReportBuilder:
    """分析报告构建器"""

    def __init__(self, output_dir: str = "./data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_report(
        self,
        title: str,
        subtitle: str = "",
        kpi_html: str = "",
        chart_sections: list = None,
        anomaly_html: str = "",
        correlation_html: str = "",
        whatif_html: str = "",
        narrative: str = "",
        data_overview: str = "",
        meta_info: str = "",
    ) -> Dict[str, Any]:
        """组合所有分析结果生成 Apple-style HTML 报告"""
        body_parts = []
        chart_sections = chart_sections or []

        # 数据概览
        if data_overview:
            body_parts.append(
                '<div class="section">'
                '<h2 class="section-title">📋 数据概览</h2>'
                f'<div class="card full"><div class="section-desc" style="margin-bottom:0">{_esc(data_overview)}</div></div>'
                '</div>'
            )

        # 可视化图表
        if chart_sections:
            charts_body = "\n".join(chart_sections)
            body_parts.append(
                '<div class="section">'
                '<h2 class="section-title">📊 可视化图表</h2>'
                f'<div class="card-grid">{charts_body}</div>'
                '</div>'
            )

        # 异常检测
        if anomaly_html:
            body_parts.append(
                '<div class="section">'
                '<h2 class="section-title">🧠 高级分析洞察</h2>'
                f'{anomaly_html}'
            )

        # 关联分析 (may be inside advanced section or standalone)
        if correlation_html:
            body_parts.append(correlation_html)

        # What-If
        if whatif_html:
            body_parts.append(whatif_html)

        # 叙述性解读
        if narrative:
            body_parts.append(
                '<div class="section">'
                '<h2 class="section-title">📋 分析总结</h2>'
                f'<div class="highlight-box">{_esc(narrative)}</div>'
                '</div>'
            )

        body_content = "\n".join(body_parts)

        # Build meta line for header
        if meta_info:
            meta_html = f'<p class="meta">{_esc(meta_info)}</p>'
        else:
            meta_html = ""

        html = REPORT_TEMPLATE.format(
            title=_esc(title),
            subtitle=_esc(subtitle or "自动生成"),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            body_content=body_content,
            stats_bar=kpi_html,
            meta_html=meta_html,
        )

        safe_name = re.sub(r"[\\/:\"*?<>|.]+", "_", title.replace(" ", "_"))[:40]
        fname = f"analysis_{safe_name}_{uuid.uuid4().hex[:6]}.html"
        filepath = self.output_dir / fname
        filepath.write_text(html, encoding="utf-8")

        logger.info(f"Report created: {fname}")

        return {
            "success": True,
            "filepath": str(filepath.absolute()),
            "filename": fname,
            "url_path": f"/reports/{fname}",
        }

    def build_chart_card(self, title: str, img_html: str) -> str:
        """构建图表卡片 — Apple-style card grid item"""
        return (
            f'<div class="card">'
            f'<h3>{_esc(title)}</h3>'
            f'{img_html}'
            f'</div>'
        )

    def build_kpi_grid(self, kpis: List[Dict[str, str]]) -> str:
        """构建 KPI 统计卡片 — Apple-style stats bar"""
        if not kpis:
            return ""

        icons = ["💰", "📈", "📦", "📊", "🎯", "📉", "🏷️", "💎"]
        # Limit to at most 4 cards for the stats bar
        cards = []
        for i, kpi in enumerate(kpis[:4]):
            icon = icons[i % len(icons)]
            card_class = "stat-card highlight" if i == 1 else "stat-card"
            cards.append(
                f'<div class="{card_class}">'
                f'<div class="icon">{icon}</div>'
                f'<div class="value">{_esc(kpi.get("value", "-"))}</div>'
                f'<div class="label">{_esc(kpi.get("label", ""))}</div>'
                f'</div>'
            )

        # Ensure we have exactly 4 cards (pad with empty if needed)
        while len(cards) < 4:
            i = len(cards)
            icon = icons[i % len(icons)]
            cards.append(
                f'<div class="stat-card">'
                f'<div class="icon">{icon}</div>'
                f'<div class="value">—</div>'
                f'<div class="label">—</div>'
                f'</div>'
            )

        return '<div class="stats-bar">\n' + "\n".join(cards) + "\n</div>"

    def build_anomaly_section(self, anomaly_result: Dict[str, Any]) -> str:
        """构建异常检测 HTML — adv-card badge-blue 样式"""
        if not anomaly_result:
            return ""

        if anomaly_result.get("error"):
            return (
                '<div class="adv-card">'
                '<span class="badge badge-red">⚠️ 异常检测</span>'
                '<h3>数据质量评估</h3>'
                f'<p>异常检测失败: {_esc(anomaly_result["error"])}</p>'
                '</div>'
            )

        rows = []
        total = anomaly_result.get("total_anomalies", 0)
        for col, col_result in anomaly_result.get("columns", {}).items():
            combined = col_result.get("combined", [])
            for a in combined[:10]:
                rd = a.get("row_data", {})
                rows.append(
                    f'<tr><td>{_esc(col)}</td><td>#{a["row_index"]}</td>'
                    f'<td>{_esc(str(rd.get(col, "?")))}</td>'
                    f'<td><span style="color:#FF9500;font-weight:600">异常</span></td></tr>'
                )

        if not rows:
            return (
                '<div class="adv-card">'
                '<span class="badge badge-blue">🔍 异常检测</span>'
                '<h3>数据质量评估</h3>'
                '<p>未检测到显著异常数据点（Z-score + IQR 双方法交集），数据整体质量良好。</p>'
                '</div>'
            )

        table_html = (
            '<table class="data-table" style="margin-top:12px">'
            '<thead><tr><th>列名</th><th>行索引</th><th>值</th><th>状态</th></tr></thead>'
            '<tbody>' + "\n".join(rows) + '</tbody>'
            '</table>'
        )

        return (
            '<div class="adv-card">'
            '<span class="badge badge-blue">🔍 异常检测</span>'
            '<h3>数据质量评估</h3>'
            f'<p>共检测到 <strong>{total}</strong> 个异常数据点（Z-score + IQR 双方法交集），'
            f'以下为部分异常记录：</p>'
            f'{table_html}'
            '</div>'
        )

    def build_correlation_section(self, corr_result: Dict[str, Any]) -> str:
        """构建相关性分析 HTML — adv-card badge-green 样式"""
        if not corr_result:
            return ""

        if corr_result.get("error"):
            return (
                '<div class="adv-card">'
                '<span class="badge badge-green">🔗 关联分析</span>'
                '<h3>数值列相关性</h3>'
                f'<p>{_esc(corr_result["error"])}</p>'
                '</div>'
            )

        pairs = corr_result.get("pairs", [])
        if not pairs:
            return (
                '<div class="adv-card">'
                '<span class="badge badge-green">🔗 关联分析</span>'
                '<h3>数值列相关性</h3>'
                '<p>无足够数值列计算相关性。</p>'
                '</div>'
            )

        rows = []
        for p in pairs:
            corr = p["correlation"]
            if abs(corr) >= 0.7:
                tag_cls = "badge-green" if corr > 0 else "badge-red"
                label = "强正相关" if corr > 0 else "强负相关"
            elif abs(corr) >= 0.4:
                tag_cls = "badge-blue"
                label = "中等正相关" if corr > 0 else "中等负相关"
            else:
                tag_cls = ""
                label = "弱相关"
            tag = f'<span class="badge {tag_cls}">{label}</span>' if tag_cls else label
            rows.append(
                f'<tr><td>{_esc(p["col1"])}</td><td>{_esc(p["col2"])}</td>'
                f'<td>{corr:.3f}</td><td>{tag}</td></tr>'
            )

        table_html = (
            '<table class="data-table" style="margin-top:12px">'
            '<thead><tr><th>列1</th><th>列2</th><th>相关系数</th><th>强度</th></tr></thead>'
            '<tbody>' + "\n".join(rows) + '</tbody>'
            '</table>'
        )

        return (
            '<div class="adv-card">'
            '<span class="badge badge-green">🔗 关联分析</span>'
            '<h3>数值列相关性矩阵</h3>'
            f'<p>共分析 {len(pairs)} 对数值列的相关性：</p>'
            f'{table_html}'
            '</div>'
        )

    def build_whatif_section(self, whatif_results: List[Dict[str, Any]]) -> str:
        """构建 What-If 模拟 HTML — adv-card badge-orange 样式"""
        if not whatif_results:
            return ""

        parts = []
        for r in whatif_results:
            if r.get("error"):
                parts.append(
                    '<div class="adv-card">'
                    '<span class="badge badge-orange">📐 场景模拟</span>'
                    f'<p>{_esc(r["error"])}</p>'
                    '</div>'
                )
                continue

            label = r.get("label", r.get("scenario", "模拟"))
            orig = r.get("original", {})
            new = r.get("new", {})
            change = r.get("change", {})

            # Build stats comparison table
            parts.append(
                '<div class="adv-card">'
                '<span class="badge badge-orange">📐 场景模拟</span>'
                f'<h3>{_esc(label)}</h3>'
                '<table class="data-table" style="margin-top:12px">'
                '<thead><tr><th>指标</th><th>原始值</th><th>变化后</th><th>变动</th></tr></thead>'
                '<tbody>'
                f'<tr><td>总和</td><td>{orig.get("sum", "-")}</td><td>{new.get("sum", "-")}</td>'
                f'<td>{change.get("sum_delta", "-")} ({change.get("sum_delta_pct", "-")}%)</td></tr>'
                f'<tr><td>均值</td><td>{orig.get("mean", "-")}</td><td>{new.get("mean", "-")}</td><td>—</td></tr>'
                '</tbody>'
                '</table>'
            )

            # Impact estimates on other columns
            impacts = r.get("impacts", {})
            if impacts:
                parts.append(
                    '<p style="margin-top:12px"><strong>对其他列的影响估算:</strong></p>'
                    '<table class="data-table" style="margin-top:8px">'
                    '<thead><tr><th>列名</th><th>原均值</th><th>预估新均值</th><th>变动%</th><th>相关系数</th></tr></thead>'
                    '<tbody>'
                )
                for col, impact in impacts.items():
                    parts.append(
                        f'<tr><td>{_esc(col)}</td><td>{_esc(str(impact["original_mean"]))}</td>'
                        f'<td>{_esc(str(impact["estimated_new_mean"]))}</td>'
                        f'<td>{_esc(str(impact["delta_pct"]))}%</td>'
                        f'<td>{_esc(str(impact["correlation"]))}</td></tr>'
                    )
                parts.append('</tbody></table>')

            parts.append('</div>')

        return "\n".join(parts)

    def build_insight_list(self, insights: List[Dict[str, str]]) -> str:
        """构建洞察列表 — Apple-style insight bullet list"""
        if not insights:
            return ""

        items = []
        colors = ["#007AFF", "#34C759", "#FF9500", "#FF3B30", "#5856D6"]
        for i, ins in enumerate(insights):
            color = colors[i % len(colors)]
            items.append(
                f'<li>'
                f'<span class="s-bullet" style="background:{color}"></span>'
                f'<div><span class="label">{_esc(ins.get("label", ""))}</span><br>'
                f'<span class="detail">{_esc(ins.get("detail", ""))}</span></div>'
                f'</li>'
            )

        return (
            '<div class="card full">'
            '<ul class="summary-bullets">'
            + "\n".join(items) +
            '</ul>'
            '</div>'
        )

    def build_data_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        title: str = "",
    ) -> str:
        """构建数据表格 — Apple-style data table"""
        header_row = "".join(f"<th>{_esc(h)}</th>" for h in headers)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{_esc(cell)}</td>" for cell in row)
            body_rows.append(f"<tr>{cells}</tr>")

        title_html = f"<h3>{_esc(title)}</h3>" if title else ""
        table_html = (
            f'{title_html}'
            f'<table class="data-table">'
            f'<thead><tr>{header_row}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody>'
            f'</table>'
        )
        return f'<div class="card full">{table_html}</div>'


# 单例
_report_builder_instance: Optional[ReportBuilder] = None


def get_report_builder() -> ReportBuilder:
    global _report_builder_instance
    if _report_builder_instance is None:
        _report_builder_instance = ReportBuilder()
    return _report_builder_instance
