"""
Dashboard MCP Server
====================
生成交互式数据仪表板 HTML 页面，组合多张图表和关键指标。

原理：使用 QuickChart.io 生成各张图表，嵌入到统一的 HTML 模板中，
形成一个完整的分析看板。支持 KPI 卡片、图表网格、标题和描述。

输出：自包含 HTML 文件，可直接在浏览器打开。
"""

import html as _html
import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from src.mcp.quickchart_server import QuickChartServer

logger = logging.getLogger(__name__)


def _esc(s: Any) -> str:
    """HTML-escape user data to prevent XSS injection."""
    return _html.escape(str(s), quote=True)

DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
    color: #e0e0e0;
    padding: 40px;
  }}
  .header {{
    text-align: center;
    margin-bottom: 40px;
    padding-bottom: 24px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }}
  .header h1 {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .header .subtitle {{
    color: #8892b0;
    font-size: 0.9rem;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 36px;
  }}
  .kpi-card {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
  }}
  .kpi-value {{
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 4px;
    background: linear-gradient(135deg, #667eea, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .kpi-label {{
    font-size: 0.85rem;
    color: #8892b0;
  }}
  .kpi-change {{
    font-size: 0.8rem;
    margin-top: 4px;
  }}
  .kpi-change.up {{ color: #4ade80; }}
  .kpi-change.down {{ color: #f87171; }}
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 24px;
    margin-bottom: 36px;
  }}
  .chart-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    overflow: hidden;
  }}
  .chart-card h3 {{
    font-size: 1rem;
    margin-bottom: 16px;
    color: #cbd5e1;
    font-weight: 600;
  }}
  .chart-card img {{
    width: 100%;
    border-radius: 8px;
  }}
  .footer {{
    text-align: center;
    color: #475569;
    font-size: 0.75rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
  }}
  .section-title {{
    font-size: 1.25rem;
    font-weight: 600;
    color: #cbd5e1;
    margin: 32px 0 16px 0;
  }}
  @media (max-width: 768px) {{
    body {{ padding: 16px; }}
    .charts-grid {{ grid-template-columns: 1fr; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <div class="subtitle">{subtitle} · 生成于 {generated_at}</div>
</div>

{kpi_section}

{charts_section}

{description_section}

<div class="footer">
  Aegis Dashboard · Powered by QuickChart.io · 数据仅供内部参考
</div>
</body>
</html>"""


class DashboardServer:
    """Dashboard MCP 服务 — 交互式数据看板生成"""

    def __init__(self, output_dir: str = "./data/dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_server = QuickChartServer()

    def create_dashboard(
        self,
        title: str,
        charts: List[Dict[str, Any]],
        kpis: Optional[List[Dict[str, str]]] = None,
        subtitle: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """
        创建数据仪表板

        Args:
            title: 仪表板标题
            charts: 图表列表，每项:
              {
                "title": "月销售额",
                "type": "bar",          # bar/line/pie/doughnut/radar
                "labels": ["1月","2月","3月"],
                "datasets": [{"label":"销售额","data":[100,150,120]}],
                "width": 800,           # 可选
                "height": 400,          # 可选
              }
            kpis: KPI 指标卡片列表，每项:
              {
                "label": "总销售额",
                "value": "¥370,000",
                "change": "+15.2%",     # 可选
                "direction": "up",      # up/down/neutral (可选)
              }
            subtitle: 副标题
            description: 底部描述文本（支持 Markdown 风格的纯文本）

        Returns:
            {"success": True, "filepath": "...", "url_path": "...", "html": "..."}
        """
        # 生成所有图表
        chart_images = []
        for i, chart_cfg in enumerate(charts):
            chart_type = chart_cfg.get("type", "bar")
            labels = chart_cfg.get("labels", [])
            datasets = chart_cfg.get("datasets", [])
            c_title = chart_cfg.get("title", f"Chart {i + 1}")
            width = chart_cfg.get("width", 800)
            height = chart_cfg.get("height", 400)

            result = self._generate_chart_by_type(
                chart_type, labels, datasets, c_title, width, height
            )
            chart_images.append({
                "title": c_title,
                "html": self.chart_server.chart_to_html_img(result, c_title),
                "success": result.get("success", False),
            })

        # 构建 KPI 区域
        kpi_html = ""
        if kpis:
            kpi_cards = []
            for kpi in kpis:
                change_html = ""
                if kpi.get("change"):
                    d = kpi.get("direction", "neutral")
                    change_html = f'<div class="kpi-change {d}">{_esc(kpi["change"])}</div>'
                kpi_cards.append(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-value">{_esc(kpi["value"])}</div>'
                    f'<div class="kpi-label">{_esc(kpi["label"])}</div>'
                    f'{change_html}'
                    f'</div>'
                )
            kpi_html = '<div class="kpi-grid">\n' + "\n".join(kpi_cards) + "\n</div>"

        # 构建图表区域
        chart_cards = []
        for c in chart_images:
            status_badge = "" if c["success"] else " ⚠️ 生成失败"
            chart_cards.append(
                f'<div class="chart-card">'
                f'<h3>{_esc(c["title"])}{status_badge}</h3>'
                f'{c["html"]}'
                f'</div>'
            )
        charts_html = ""
        if chart_cards:
            charts_html = '<div class="section-title">📊 数据图表</div>\n<div class="charts-grid">\n' + "\n".join(chart_cards) + "\n</div>"

        # 描述区域
        desc_html = ""
        if description:
            desc_html = (
                f'<div class="section-title">📋 分析说明</div>'
                f'<div style="color:#94a3b8;line-height:1.8;max-width:800px;margin:0 auto;">'
                f'{_esc(description)}'
                f'</div>'
            )

        # 渲染模板
        html = DASHBOARD_TEMPLATE.format(
            title=_esc(title),
            subtitle=_esc(subtitle or "数据分析仪表板"),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            kpi_section=kpi_html,
            charts_section=charts_html,
            description_section=desc_html,
        )

        # 保存到文件
        safe_name = title.replace(" ", "_").replace("/", "_")[:40]
        fname = f"dashboard_{safe_name}_{uuid.uuid4().hex[:6]}.html"
        fpath = self.output_dir / fname
        fpath.write_text(html, encoding="utf-8")

        logger.info(f"Dashboard created: {fname} ({len(charts)} charts, {len(kpis or [])} KPIs)")

        return {
            "success": True,
            "filepath": str(fpath.absolute()),
            "url_path": f"/dashboards/{fname}",
            "chart_count": len(charts),
            "kpi_count": len(kpis or []),
            "charts_success": sum(1 for c in chart_images if c["success"]),
        }

    def _generate_chart_by_type(
        self,
        chart_type: str,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: str,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        """根据类型调用对应的图表生成方法"""
        if chart_type == "bar":
            return self.chart_server.bar_chart(labels, datasets, title, width=width, height=height)
        elif chart_type == "line":
            return self.chart_server.line_chart(labels, datasets, title, width=width, height=height)
        elif chart_type == "pie":
            data = datasets[0].get("data", []) if datasets else []
            return self.chart_server.pie_chart(labels, data, title, width=width, height=height)
        elif chart_type == "doughnut":
            data = datasets[0].get("data", []) if datasets else []
            return self.chart_server.doughnut_chart(labels, data, title, width=width, height=height)
        elif chart_type == "radar":
            return self.chart_server.radar_chart(labels, datasets, title, width=width, height=height)
        else:
            # 默认柱状图
            return self.chart_server.bar_chart(labels, datasets, title, width=width, height=height)


# 单例
_dashboard_instance: Optional[DashboardServer] = None


def get_dashboard_server() -> DashboardServer:
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = DashboardServer()
    return _dashboard_instance
