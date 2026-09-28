"""第四步：输出结果 —— 控制台表格 + JSON 报告。"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path


def display_width(text: str) -> int:
    """中文字符占 2 个显示宽度，用它来对齐表格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width: int, align: str = "left") -> str:
    text = str(text)
    space = max(0, width - display_width(text))
    if align == "right":
        return " " * space + text
    if align == "center":
        left = space // 2
        return " " * left + text + " " * (space - left)
    return text + " " * space


COLUMNS = [
    ("区域", 6, "left"),
    ("位置(x,y)", 13, "left"),
    ("面积占比", 10, "right"),
    ("绿色占比", 10, "right"),
    ("黄化占比", 10, "right"),
    ("病斑数", 8, "right"),
    ("健康分", 8, "right"),
    ("判定", 18, "left"),
]


def format_table(regions) -> str:
    header = "  ".join(pad(name, width, "center") for name, width, _ in COLUMNS)
    lines = [header, "-" * display_width(header)]
    for region in regions:
        row = [
            (f"#{region.index}", 6, "left"),
            (f"{region.x},{region.y}", 13, "left"),
            (f"{region.area_ratio * 100:.1f}%", 10, "right"),
            (f"{region.green_ratio * 100:.1f}%", 10, "right"),
            (f"{region.yellow_ratio * 100:.1f}%", 10, "right"),
            (region.spot_count, 8, "right"),
            (region.health_score, 8, "right"),
            (region.label, 18, "left"),
        ]
        lines.append("  ".join(pad(text, width, align) for text, width, align in row))
    return "\n".join(lines)


def format_console(result) -> str:
    """把一次检测结果排版成人类可读的控制台文本。"""
    s = result.summary
    lines = [
        "=" * 62,
        f"图像：{result.name}",
        f"尺寸：{result.width} x {result.height}",
        "-" * 62,
    ]
    if s["region_count"] == 0:
        lines.append("结果：未检测到植物（植被覆盖度 0%）")
        lines.append("=" * 62)
        return "\n".join(lines)

    lines.append(
        f"植被覆盖度：{s['coverage_ratio'] * 100:5.1f}%    "
        f"估计植株数：{s['estimated_plant_count']}"
    )
    lines.append(
        f"总体健康分：{s['mean_health_score']:3d}（{s['overall_label']}）    "
        f"疑似病斑：{s['total_spot_count']} 处"
    )
    lines.append(
        f"像素构成：健康绿 {s['green_ratio'] * 100:.1f}% | "
        f"黄化 {s['yellow_ratio'] * 100:.1f}% | "
        f"病斑/坏死 {s['spot_ratio'] * 100:.1f}%"
    )
    lines.append("-" * 62)
    lines.append(format_table(result.regions))
    lines.append("=" * 62)
    return "\n".join(lines)


def save_json(result, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": result.name,
        "image_size": {"width": result.width, "height": result.height},
        "segment_config": result.segment_config,
        "analyze_config": result.analyze_config,
        "segment_info": result.segment_info,
        "summary": result.summary,
        "regions": [r.to_dict() for r in result.regions],
        "outputs": result.outputs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
