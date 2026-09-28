"""第二步：对分割出的植物区域做统计与健康评估。

对每个连通域（一株植物 / 一簇叶片）统计：
面积、外接框、绿色占比、黄化占比、病斑数量、健康分与判定结论。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class AnalyzeConfig:
    """区域分析与健康评估参数。"""

    # 健康绿色：正常叶片的范围
    green_hue_min: int = 30
    green_hue_max: int = 95
    green_sat_min: int = 50
    green_val_min: int = 40

    # 黄化：偏黄偏橙，说明缺素 / 缺水 / 叶片衰老
    yellow_hue_min: int = 12
    yellow_hue_max: int = 30
    yellow_sat_min: int = 60
    yellow_val_min: int = 80

    # 病斑：区域内既不是健康绿也不是黄化，且明显偏暗（褐斑 / 黑斑 / 干枯）
    spot_val_max: int = 110
    spot_min_area_ratio: float = 0.0004   # 单个病斑相对所在区域面积的最小值
    notable_region_ratio: float = 0.02    # 相对植株总面积，算作"一株"的门槛

    # 健康分等级线
    healthy_score: int = 80
    mild_score: int = 60
    stress_score: int = 40

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Region:
    """一个植物连通域的检测结果。"""

    index: int
    x: int
    y: int
    w: int
    h: int
    area: int
    area_ratio: float          # 占植株总面积的比例
    coverage_ratio: float      # 占整幅图像的比例
    green_ratio: float
    yellow_ratio: float
    spot_ratio: float
    spot_count: int
    mean_exg: float
    health_score: int
    label: str

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def to_dict(self) -> dict:
        return asdict(self)


def classify_pixels(
    mask: np.ndarray, hsv: np.ndarray, exg: np.ndarray, cfg: AnalyzeConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在植被掩膜内，把像素分成「健康绿 / 黄化 / 疑似病斑」三类。"""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    plant = mask > 0

    green = (
        (h >= cfg.green_hue_min)
        & (h <= cfg.green_hue_max)
        & (s >= cfg.green_sat_min)
        & (v >= cfg.green_val_min)
    )
    yellow = (
        (h >= cfg.yellow_hue_min)
        & (h < cfg.yellow_hue_max)
        & (s >= cfg.yellow_sat_min)
        & (v >= cfg.yellow_val_min)
    )
    spot = (v <= cfg.spot_val_max) | (exg < 0.0)

    green &= plant
    yellow &= plant & ~green
    spot &= plant & ~green & ~yellow
    # 开运算去掉叶片边缘 1~2 像素的抗锯齿伪影，避免把叶缘当成病斑
    spot_u8 = cv2.morphologyEx(
        spot.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    spot = spot_u8 > 0
    return green, yellow, spot


def count_blobs(binary: np.ndarray, min_area: float) -> int:
    """统计二值图中面积达标的斑块个数。"""
    if not binary.any():
        return 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    return int(
        sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area)
    )


def health_label(score: int, cfg: AnalyzeConfig) -> str:
    if score >= cfg.healthy_score:
        return "健康"
    if score >= cfg.mild_score:
        return "轻微黄化/亚健康"
    if score >= cfg.stress_score:
        return "明显胁迫"
    return "疑似病害"


def health_score(
    green_ratio: float,
    yellow_ratio: float,
    spot_ratio: float,
    spot_count: int,
) -> int:
    """健康分 = 100 分制加权，病斑额外扣分，范围 0~100。"""
    raw = 100.0 * (green_ratio + 0.5 * yellow_ratio + 0.15 * spot_ratio)
    penalty = min(12.0, spot_count * 0.8)
    return int(np.clip(round(raw - penalty), 0, 100))


def analyze_regions(
    mask: np.ndarray,
    hsv: np.ndarray,
    exg: np.ndarray,
    cfg: AnalyzeConfig | None = None,
) -> tuple[list[Region], dict]:
    """对植被掩膜做连通域分析，返回 (区域列表, 全图汇总)。"""
    cfg = cfg or AnalyzeConfig()
    total_px = mask.shape[0] * mask.shape[1]
    plant_px = int(np.count_nonzero(mask))

    if plant_px == 0:
        return [], {
            "plant_pixels": 0,
            "coverage_ratio": 0.0,
            "region_count": 0,
            "notable_region_count": 0,
            "estimated_plant_count": 0,
            "largest_region_ratio": 0.0,
            "mean_health_score": 0,
            "overall_label": "未检测到植物",
            "health_distribution": {},
            "green_ratio": 0.0,
            "yellow_ratio": 0.0,
            "spot_ratio": 0.0,
            "total_spot_count": 0,
        }

    green, yellow, spot = classify_pixels(mask, hsv, exg, cfg)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    regions: list[Region] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        blob = labels == label
        g_ratio = float(np.count_nonzero(green & blob)) / area
        y_ratio = float(np.count_nonzero(yellow & blob)) / area
        s_ratio = float(np.count_nonzero(spot & blob)) / area

        spot_mask = (spot & blob).astype(np.uint8)
        n_spot = count_blobs(spot_mask, max(3.0, cfg.spot_min_area_ratio * area))
        score = health_score(g_ratio, y_ratio, s_ratio, n_spot)

        regions.append(
            Region(
                index=0,  # 稍后按面积排序再统一编号
                x=int(stats[label, cv2.CC_STAT_LEFT]),
                y=int(stats[label, cv2.CC_STAT_TOP]),
                w=int(stats[label, cv2.CC_STAT_WIDTH]),
                h=int(stats[label, cv2.CC_STAT_HEIGHT]),
                area=area,
                area_ratio=area / plant_px,
                coverage_ratio=area / total_px,
                green_ratio=round(g_ratio, 4),
                yellow_ratio=round(y_ratio, 4),
                spot_ratio=round(s_ratio, 4),
                spot_count=n_spot,
                mean_exg=round(float(exg[blob].mean()), 4),
                health_score=score,
                label=health_label(score, cfg),
            )
        )

    regions.sort(key=lambda r: r.area, reverse=True)
    for i, region in enumerate(regions, start=1):
        region.index = i

    notable = [r for r in regions if r.area_ratio >= cfg.notable_region_ratio]
    weights = np.array([r.area for r in regions], dtype=np.float64)
    mean_score = int(
        round(float(np.average([r.health_score for r in regions], weights=weights)))
    )

    distribution: dict[str, int] = {}
    for region in regions:
        distribution[region.label] = distribution.get(region.label, 0) + 1

    summary = {
        "plant_pixels": plant_px,
        "coverage_ratio": round(plant_px / total_px, 4),      # 植被覆盖度
        "region_count": len(regions),                          # 全部连通域
        "notable_region_count": len(notable),                  # 面积达标的区域
        "estimated_plant_count": len(notable),                 # 估计植株数
        "largest_region_ratio": round(regions[0].area_ratio, 4) if regions else 0.0,
        "mean_health_score": mean_score,
        "overall_label": health_label(mean_score, cfg),
        "health_distribution": distribution,
        "green_ratio": round(float(np.count_nonzero(green)) / plant_px, 4),
        "yellow_ratio": round(float(np.count_nonzero(yellow)) / plant_px, 4),
        "spot_ratio": round(float(np.count_nonzero(spot)) / plant_px, 4),
        "total_spot_count": int(sum(r.spot_count for r in regions)),
    }
    return regions, summary
