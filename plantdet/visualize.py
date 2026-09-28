"""第三步：把检测结果画出来（标注图 + 三联对比图）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MASK_COLOR = (60, 210, 70)    # BGR 绿色：叠加植被区域
SPOT_COLOR = (40, 40, 235)    # BGR 红色：叠加疑似病斑
BOX_COLOR = (255, 210, 40)    # BGR 天蓝：植株外接框


def overlay_masks(
    bgr: np.ndarray,
    mask: np.ndarray,
    spot_mask: np.ndarray | None = None,
    alpha: float = 0.38,
) -> np.ndarray:
    """把植被掩膜（绿）与病斑掩膜（红）半透明叠在原图上。"""
    canvas = bgr[..., :3].copy().astype(np.float32)

    layer = canvas.copy()
    layer[mask > 0] = MASK_COLOR
    canvas = canvas * (1 - alpha) + layer * alpha

    if spot_mask is not None and spot_mask.any():
        layer = canvas.copy()
        layer[spot_mask > 0] = SPOT_COLOR
        canvas = canvas * (1 - alpha) + layer * alpha

    return np.clip(canvas, 0, 255).astype(np.uint8)


def draw_regions(bgr: np.ndarray, regions, show_score: bool = True) -> np.ndarray:
    """在图上画外接框 + 编号 + 健康分。"""
    canvas = bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BOX_COLOR, 2)

        text = f"#{region.index}"
        if show_score:
            text += f" {region.label} {region.health_score}"
        scale, thickness = 0.5, 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        ty = max(y - 6, th + 8)
        cv2.rectangle(
            canvas,
            (x, ty - th - 6),
            (x + tw + 8, ty + 4),
            (25, 25, 25),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            canvas,
            text,
            (x + 4, ty - 1),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def _setup_chinese_font() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def make_composite(
    bgr: np.ndarray,
    mask: np.ndarray,
    annotated: np.ndarray,
    summary: dict,
    title: str = "",
    save_to: str | Path | None = None,
):
    """生成「原图 / 植被掩膜 / 检测结果」三联图，适合放进报告。"""
    _setup_chinese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    panels = [
        (cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB), "1. 原图"),
        (mask, "2. 植被掩膜 ExG + HSV\n（白色 = 检测到的植物）"),
        (
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            "3. 检测结果\n（绿 = 植物，红 = 疑似病斑）",
        ),
    ]
    for ax, (image, subtitle) in zip(axes, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(subtitle, fontsize=11)
        ax.axis("off")

    headline = (
        f"覆盖度 {summary['coverage_ratio'] * 100:.1f}%    "
        f"估计植株数 {summary['estimated_plant_count']}    "
        f"总体健康分 {summary['mean_health_score']}（{summary['overall_label']}）    "
        f"疑似病斑 {summary['total_spot_count']} 处"
    )
    fig.suptitle(f"{title}\n{headline}" if title else headline, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig
