"""把分割、分析、可视化、报告串成一条流水线。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .analyze import AnalyzeConfig, Region, analyze_regions, classify_pixels
from .report import save_json
from .segment import SegmentConfig, compute_exg, hsv_of, vegetation_mask
from .visualize import draw_regions, make_composite, overlay_masks


@dataclass
class Result:
    """一次检测的完整结果。"""

    name: str
    width: int
    height: int
    regions: list[Region] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    segment_info: dict = field(default_factory=dict)
    segment_config: dict = field(default_factory=dict)
    analyze_config: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    annotated: np.ndarray | None = None


def analyze_image(
    bgr: np.ndarray,
    name: str = "image",
    segment_cfg: SegmentConfig | None = None,
    analyze_cfg: AnalyzeConfig | None = None,
) -> Result:
    """对一张 BGR 图像做完整分析（不落盘）。"""
    segment_cfg = segment_cfg or SegmentConfig()
    analyze_cfg = analyze_cfg or AnalyzeConfig()
    bgr = np.ascontiguousarray(bgr)

    mask, segment_info = vegetation_mask(bgr, segment_cfg)
    hsv = hsv_of(bgr)
    exg = compute_exg(bgr)
    regions, summary = analyze_regions(mask, hsv, exg, analyze_cfg)

    _, _, spot = classify_pixels(mask, hsv, exg, analyze_cfg)
    overlay = overlay_masks(bgr, mask, spot)
    annotated = draw_regions(overlay, regions)

    height, width = bgr.shape[:2]
    return Result(
        name=name,
        width=width,
        height=height,
        regions=regions,
        summary=summary,
        segment_info=segment_info,
        segment_config=segment_cfg.to_dict(),
        analyze_config=analyze_cfg.to_dict(),
        image=bgr,
        mask=mask,
        annotated=annotated,
    )


def analyze_path(
    image_path: str | Path,
    out_dir: str | Path = "results",
    segment_cfg: SegmentConfig | None = None,
    analyze_cfg: AnalyzeConfig | None = None,
    write_figure: bool = True,
    figure_title: str = "",
) -> Result:
    """读一张图 -> 分析 -> 保存标注图 / 掩膜图 / JSON 报告。"""
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    bgr = read_image(image_path)

    result = analyze_image(
        bgr,
        name=image_path.stem,
        segment_cfg=segment_cfg,
        analyze_cfg=analyze_cfg,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    annotated_path = out_dir / f"{stem}_detected.png"
    mask_path = out_dir / f"{stem}_mask.png"
    json_path = out_dir / f"{stem}_report.json"

    write_image(annotated_path, result.annotated)
    write_image(mask_path, result.mask)
    result.outputs = {
        "annotated": str(annotated_path),
        "mask": str(mask_path),
    }
    save_json(result, json_path)
    result.outputs["report"] = str(json_path)

    if write_figure:
        figure_path = out_dir / f"{stem}_figure.png"
        figure = make_composite(
            result.image,
            result.mask,
            result.annotated,
            result.summary,
            title=figure_title or image_path.name,
        )
        figure.savefig(figure_path)
        result.outputs["figure"] = str(figure_path)

    return result


def read_image(path: str | Path) -> np.ndarray:
    """读取图片，顺便处理中文路径和带透明通道的 PNG。"""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"无法读取图片：{path}")

    if bgr.ndim == 3 and bgr.shape[2] == 4:
        alpha = bgr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = bgr[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        bgr = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(bgr[:, :, :3])


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """写图。走 imencode + tofile，避免 cv2.imwrite 在中文路径下写出乱码文件名。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    buffer.tofile(str(path))
    return path
