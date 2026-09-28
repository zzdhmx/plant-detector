"""第一步：把画面里的植物从背景中分割出来。

思路（经典的 ExG 植被指数法）：
1. 把像素转换到色度坐标，计算超绿指数 ExG = 2g - r - b，绿色越强值越大；
2. 用 Otsu 自动阈值在 ExG 图上切一刀，得到"像植物"的像素；
3. 再用 HSV 颜色区间做一次过滤，去掉灰白色、高亮反光等误检；
4. 形态学开运算去噪点、闭运算补缝隙，最后删掉过小的碎块。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class SegmentConfig:
    """植被分割参数。所有参数都可以在命令行里覆盖。"""

    # HSV 粗筛：正常绿色在 OpenCV 的 H 通道里大约落在 25~95
    hue_min: int = 25
    hue_max: int = 95
    sat_min: int = 40          # 饱和度下限，滤掉灰白区域
    val_min: int = 30          # 亮度下限，滤掉阴影和黑边

    # 发黄、失绿的叶片颜色已经偏橙黄，用单独一组阈值放宽接纳，
    # 但必须"挨着绿色主体"才算植物——否则土块、枯枝、木栅栏都会被误检。
    yellow_hue_min: int = 12
    yellow_hue_max: int = 25
    yellow_sat_min: int = 60
    yellow_val_min: int = 60
    # 黄叶与绿色主体之间的"接触容差"：只要一片黄叶有像素落在绿色主体的这个
    # 范围内，整片黄叶都会被接纳（而不是只接纳靠近的那一小圈）。0 = 不接纳黄叶。
    yellow_connect_px: int = 15

    exg_bias: float = 0.0      # 手动微调 Otsu 阈值，正数=更严格
    open_ksize: int = 3        # 开运算核大小
    close_ksize: int = 7       # 闭运算核大小
    min_area_ratio: float = 0.0005   # 小于该比例的碎块直接丢弃
    fill_holes: bool = True          # 填充叶片内部的小孔洞

    def to_dict(self) -> dict:
        return asdict(self)


def compute_exg(bgr: np.ndarray) -> np.ndarray:
    """计算超绿指数 ExG = 2g - r - b（色度归一化版本，范围约 -1 ~ 1）。

    归一化的好处：不容易被整体明暗影响，阴天拍的图也能用。
    """
    img = bgr[..., :3].astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    total = b + g + r + 1e-6
    return 2.0 * (g / total) - (r / total) - (b / total)


def to_uint8(values: np.ndarray) -> np.ndarray:
    """把任意范围的浮点图拉伸到 0~255，方便交给 OpenCV 处理。"""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - lo) / (hi - lo) * 255.0).astype(np.uint8)


def otsu_threshold(values_u8: np.ndarray) -> int:
    """Otsu 自动阈值：让前景和背景两类内部的方差之和最小。"""
    threshold, _ = cv2.threshold(
        values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return int(threshold)


def hsv_of(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2HSV)


def vegetation_mask(
    bgr: np.ndarray, cfg: SegmentConfig | None = None
) -> tuple[np.ndarray, dict]:
    """返回 (植被掩膜, 诊断信息)。掩膜为 uint8，前景=255。

    掩膜 = 绿色主体（ExG 超阈值 且 HSV 落在绿色区间）
          + 紧贴绿色主体的黄化叶片（同样要过 ExG 这一关）。
    """
    cfg = cfg or SegmentConfig()
    exg = compute_exg(bgr)
    exg_u8 = to_uint8(exg)
    thr = otsu_threshold(exg_u8)
    effective = int(np.clip(thr + cfg.exg_bias * 255.0 / 100.0, 0, 255))

    hsv = hsv_of(bgr)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    exg_ok = exg_u8 > effective

    green = (
        (h >= cfg.hue_min)
        & (h <= cfg.hue_max)
        & (s >= cfg.sat_min)
        & (v >= cfg.val_min)
        & exg_ok
    )
    yellow = (
        (h >= cfg.yellow_hue_min)
        & (h < cfg.yellow_hue_max)
        & (s >= cfg.yellow_sat_min)
        & (v >= cfg.yellow_val_min)
        & exg_ok
    )

    accepted_yellow = np.zeros_like(green)
    # 先清理绿色主体：开运算会把土壤噪点产生的零星"假绿色"直接抹掉，
    # 只有成形的绿色区域才有资格"吸附"周围的黄叶。
    green_core = clean_mask(green.astype(np.uint8) * 255, cfg)

    if cfg.yellow_connect_px > 0 and yellow.any() and green_core.any():
        ksize = cfg.yellow_connect_px | 1  # 保证是奇数
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        near_green = cv2.dilate(green_core, k) > 0
        # 以整片黄叶为单位判断：只要有一小块贴着绿色主体，整片都算植物。
        _, labels, _, _ = cv2.connectedComponentsWithStats(
            yellow.astype(np.uint8) * 255, connectivity=8
        )
        touched = np.unique(labels[near_green & (labels > 0)])
        if touched.size:
            accepted_yellow = yellow & np.isin(labels, touched)

    mask = (green_core | (accepted_yellow.astype(np.uint8) * 255))
    mask = clean_mask(mask, cfg)

    info = {
        "exg_otsu_threshold": thr,
        "exg_effective_threshold": effective,
        "green_pixels": int(np.count_nonzero(green)),
        "yellow_pixels": int(np.count_nonzero(yellow)),
        "yellow_accepted_pixels": int(np.count_nonzero(accepted_yellow)),
        "yellow_rejected_pixels": int(np.count_nonzero(yellow & ~accepted_yellow)),
        "mask_pixels": int(np.count_nonzero(mask)),
    }
    return mask, info


def clean_mask(mask: np.ndarray, cfg: SegmentConfig | None = None) -> np.ndarray:
    """形态学清理 + 去小块 + 补洞。"""
    cfg = cfg or SegmentConfig()
    cleaned = mask.copy()

    if cfg.open_ksize > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)
    if cfg.close_ksize > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.close_ksize, cfg.close_ksize)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)

    cleaned = remove_small_objects(cleaned, cfg.min_area_ratio)
    if cfg.fill_holes:
        cleaned = fill_holes(cleaned)
    return cleaned


def remove_small_objects(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    """删除面积过小的连通域（多半是噪点或零散杂草）。"""
    if min_area_ratio <= 0:
        return mask
    total = mask.shape[0] * mask.shape[1]
    min_area = max(4.0, min_area_ratio * total)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 255
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """用外轮廓重绘前景，把叶片中间被漏掉的像素补回来。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled
