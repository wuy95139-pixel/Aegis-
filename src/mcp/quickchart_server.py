"""
QuickChart MCP Server
====================
通过 QuickChart.io API 生成图表（柱状图、折线图、饼图、雷达图等）。

原理：将数据构建为 Chart.js 配置，POST 到 https://quickchart.io/chart，
返回 PNG 图片 URL 或 base64 编码，可直接嵌入报告或 HTML 中。

不需要 API Key，免费使用（有速率限制）。
"""

import json
import logging
import urllib.request
import urllib.error
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

QUICKCHART_URL = "https://quickchart.io/chart"


class QuickChartServer:
    """QuickChart MCP 服务 — 即时图表生成"""

    def __init__(self, output_dir: str = "./data/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 便捷方法 =====

    def bar_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        width: int = 600,
        height: int = 350,
    ) -> Dict[str, Any]:
        """生成柱状图"""
        config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": datasets,
            },
            "options": {
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                },
                "scales": {
                    "x": {"title": {"display": bool(x_label), "text": x_label}},
                    "y": {"title": {"display": bool(y_label), "text": y_label}, "beginAtZero": True},
                },
            },
        }
        return self.generate_chart(config, width, height)

    def line_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        width: int = 600,
        height: int = 350,
    ) -> Dict[str, Any]:
        """生成折线图"""
        config = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": datasets,
            },
            "options": {
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                },
                "scales": {
                    "x": {"title": {"display": bool(x_label), "text": x_label}},
                    "y": {"title": {"display": bool(y_label), "text": y_label}, "beginAtZero": True},
                },
            },
        }
        return self.generate_chart(config, width, height)

    def pie_chart(
        self,
        labels: List[str],
        data: List[float],
        title: str = "",
        width: int = 450,
        height: int = 300,
    ) -> Dict[str, Any]:
        """生成饼图"""
        config = {
            "type": "pie",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0",
                        "#9966FF", "#FF9F40", "#7BC8A4", "#E84351",
                        "#5D9CEC", "#F6BB42",
                    ],
                }],
            },
            "options": {
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                },
            },
        }
        return self.generate_chart(config, width, height)

    def doughnut_chart(
        self,
        labels: List[str],
        data: List[float],
        title: str = "",
        width: int = 450,
        height: int = 300,
    ) -> Dict[str, Any]:
        """生成环形图"""
        config = {
            "type": "doughnut",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0",
                        "#9966FF", "#FF9F40", "#7BC8A4", "#E84351",
                    ],
                }],
            },
            "options": {
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                },
            },
        }
        return self.generate_chart(config, width, height)

    def radar_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: str = "",
        width: int = 450,
        height: int = 350,
    ) -> Dict[str, Any]:
        """生成雷达图"""
        config = {
            "type": "radar",
            "data": {
                "labels": labels,
                "datasets": datasets,
            },
            "options": {
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                },
            },
        }
        return self.generate_chart(config, width, height)

    # ===== 核心方法 =====

    def generate_chart(
        self,
        config: Dict[str, Any],
        width: int = 600,
        height: int = 350,
        format: str = "png",
    ) -> Dict[str, Any]:
        """
        调用 QuickChart.io API 生成图表

        Args:
            config: Chart.js 配置 (type, data, options)
            width: 图片宽度
            height: 图片高度
            format: 输出格式 (png / svg / webp)

        Returns:
            {"success": True, "url": "...", "base64": "...", "filepath": "..."}
        """
        payload = {
            "chart": config,
            "width": width,
            "height": height,
            "format": format,
            "backgroundColor": "#ffffff",
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                QUICKCHART_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                image_data = resp.read()

            # 保存到本地
            import uuid
            fname = f"chart_{uuid.uuid4().hex[:8]}.{format}"
            fpath = self.output_dir / fname
            fpath.write_bytes(image_data)

            b64 = base64.b64encode(image_data).decode("ascii")
            url = f"https://quickchart.io/chart?c={urllib.request.quote(json.dumps(config))}"

            logger.info(f"Chart generated: {fname} ({len(image_data)} bytes)")

            return {
                "success": True,
                "url": url,
                "base64": b64,
                "filepath": str(fpath.absolute()),
                "format": format,
                "width": width,
                "height": height,
            }

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")[:500]
            logger.error(f"QuickChart API error: {e.code} - {error_body}")
            return {"success": False, "error": f"HTTP {e.code}: {error_body}"}
        except Exception as e:
            logger.error(f"QuickChart generate failed: {e}")
            return {"success": False, "error": str(e)}

    def chart_to_html_img(self, chart_result: Dict[str, Any], alt: str = "Chart") -> str:
        """将图表结果转为 HTML img 标签"""
        if chart_result.get("success"):
            return f'<img src="data:image/png;base64,{chart_result["base64"]}" alt="{alt}" style="max-width:480px;width:100%;height:auto;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);" />'
        return f"<!-- Chart failed: {chart_result.get('error', 'unknown')} -->"


# 单例
_quickchart_instance: Optional[QuickChartServer] = None


def get_quickchart_server() -> QuickChartServer:
    global _quickchart_instance
    if _quickchart_instance is None:
        _quickchart_instance = QuickChartServer()
    return _quickchart_instance
