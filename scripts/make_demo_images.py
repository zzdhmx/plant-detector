"""生成演示用的合成图片。

没有现成的植物照片也能跑通整个项目：这个脚本用几何形状 + 噪声
合成 4 张测试图（健康植株 / 多株幼苗 / 病害叶片 / 没有植物的场景）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 720, 540


def hsv_to_bgr(h: float, s: float, v: float) -> np.ndarray:
    """单个 HSV 颜色（OpenCV 尺度）转 BGR。"""
    px = np.uint8([[[np.clip(h, 0, 179), np.clip(s, 0, 255), np.clip(v, 0, 255)]]])
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def bgr_tuple(h: float, s: float, v: float) -> tuple[int, int, int]:
    """给 cv2 画图函数用的整数 BGR 三元组。"""
    b, g, r = hsv_to_bgr(h, s, v)
    return int(b), int(g), int(r)


def soil_background(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """粗糙的土壤背景：棕色噪点 + 小石子 + 暗角。"""
    base = np.zeros((height, width, 3), np.float32)
    grain = rng.normal(0, 12, (height, width, 1))
    soil = hsv_to_bgr(14, 105, 118)
    base[:] = soil
    base += grain

    # 大颗粒的明暗斑块，模拟土壤颗粒感
    blotch = rng.normal(0, 1, (height // 18, width // 18)).astype(np.float32)
    blotch = cv2.resize(blotch, (width, height), interpolation=cv2.INTER_CUBIC) * 14
    base += blotch[:, :, None]

    # 小石子
    for _ in range(rng.integers(18, 34)):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(2, 7))
        tone = float(rng.uniform(120, 190))
        cv2.circle(base, (cx, cy), r, (tone, tone * 0.98, tone * 0.92), -1)
        cv2.circle(base, (cx - 1, cy - 1), max(1, r // 2), (tone + 22,) * 3, -1)

    # 暗角，让画面更像真照片
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2, height / 2
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.28 * np.clip(dist - 0.35, 0, None), 0.6, 1.0)
    base *= vignette[:, :, None]
    return np.clip(base, 0, 255)


def leaf_geometry(
    height: int,
    width: int,
    center: tuple[float, float],
    length: float,
    width_px: float,
    angle_deg: float,
    serration: float = 0.07,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回叶片掩膜以及归一化坐标 (t 沿主脉, v 垂直主脉)。"""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx - center[0]
    dy = yy - center[1]
    angle = np.deg2rad(angle_deg)
    ca, sa = np.cos(angle), np.sin(angle)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    t = (u + length / 2.0) / max(length, 1e-3)
    t_clipped = np.clip(t, 0, 1)
    half = width_px / 2.0 * np.sin(np.pi * t_clipped) ** 0.62
    half *= 1.0 - serration + serration * np.cos(t_clipped * np.pi * 14)
    inside = (t >= 0) & (t <= 1) & (np.abs(v) <= half)
    return inside, t_clipped, v


def paint_leaf(
    canvas: np.ndarray,
    mask: np.ndarray,
    t: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    hue: float,
    sat: float,
    val: float,
    leaf_width: float,
) -> None:
    """按掩膜给叶片上色：主脉、侧脉、内部明暗变化。"""
    color = hsv_to_bgr(hue, sat, val)
    shade = 1.0 + 0.18 * np.cos((t - 0.6) * np.pi) - 0.10 * np.abs(v) / max(v.max(), 1e-3)
    pixels = color[None, None, :] * shade[:, :, None]
    pixels += rng.normal(0, 6, pixels.shape)

    # 主脉（沿 v=0 的一条浅色带）
    midrib = np.abs(v) < max(1.0, 0.06 * leaf_width)
    pixels[midrib] *= 0.72

    # 侧脉：一组斜向的浅色条纹（幅度很小，只做质感）
    side = np.sin((t * 26 + np.abs(v) / max(2.0, leaf_width / 5)) * np.pi)
    veins = (side > 0.975) & (np.abs(v) > 2)
    pixels[veins] *= 0.93

    canvas[mask] = np.clip(pixels[mask], 0, 255)


def add_spots(
    canvas: np.ndarray,
    leaf_mask: np.ndarray,
    rng: np.random.Generator,
    count: int,
    radius_range: tuple[int, int] = (3, 9),
) -> int:
    """在叶片上画病斑：褐色斑心 + 黄色晕圈。返回病斑数量。"""
    ys, xs = np.nonzero(leaf_mask)
    if len(ys) == 0 or count <= 0:
        return 0

    drawn = 0
    for _ in range(count):
        pick = int(rng.integers(0, len(ys)))
        cy, cx = int(ys[pick]), int(xs[pick])
        r = int(rng.integers(radius_range[0], radius_range[1] + 1))
        angle = float(rng.uniform(0, 180))
        axes = (r, max(2, int(r * rng.uniform(0.6, 1.0))))

        halo = np.zeros_like(leaf_mask, dtype=np.uint8)
        cv2.ellipse(halo, (cx, cy), (axes[0] + 2, axes[1] + 2), angle, 0, 360, 255, -1)
        halo = (halo > 0) & leaf_mask
        canvas[halo] = canvas[halo] * 0.55 + hsv_to_bgr(24, 190, 210) * 0.45

        core = np.zeros_like(leaf_mask, dtype=np.uint8)
        cv2.ellipse(core, (cx, cy), axes, angle, 0, 360, 255, -1)
        core = (core > 0) & leaf_mask
        tone = float(rng.uniform(45, 85))
        canvas[core] = hsv_to_bgr(12, 150, tone)
        drawn += 1
    return drawn


def draw_plant(
    canvas: np.ndarray,
    rng: np.random.Generator,
    base: tuple[float, float],
    scale: float,
    leaf_count: int,
    hue: float = 52.0,
    sat: float = 155.0,
    val: float = 150.0,
    spots: int = 0,
    yellow_leaves: int = 0,
    pitch_range: tuple[float, float] = (20.0, 45.0),
) -> int:
    """在画布上画一株植物（茎 + 若干叶片），返回病斑数。"""
    h, w = canvas.shape[:2]
    x0, y0 = base
    stem_top = y0 - scale * 0.85

    # 阴影
    shadow = np.zeros((h, w), np.uint8)
    cv2.ellipse(
        shadow,
        (int(x0), int(y0 + 6)),
        (int(scale * 0.55), int(scale * 0.12)),
        0,
        0,
        360,
        255,
        -1,
    )
    shadow_mask = shadow > 0
    canvas[shadow_mask] *= 0.82

    # 茎
    cv2.line(
        canvas,
        (int(x0), int(y0)),
        (int(x0 + rng.normal(0, scale * 0.02)), int(stem_top)),
        bgr_tuple(30, 170, 120),
        max(2, int(scale * 0.035)),
        cv2.LINE_AA,
    )

    spot_total = 0
    for i in range(leaf_count):
        frac = (i + 0.6) / (leaf_count + 0.4)
        cy = y0 - (y0 - stem_top) * frac
        direction = 1 if i % 2 == 0 else -1
        length = scale * float(rng.uniform(0.75, 1.05)) * (1.0 - 0.35 * frac)
        leaf_width = length * float(rng.uniform(0.42, 0.55))
        # 叶轴方向：向外展开并略微上扬，再转成角度（图像坐标 y 向下）
        pitch = np.deg2rad(float(rng.uniform(*pitch_range)))
        dir_x = direction * np.cos(pitch)
        dir_y = -np.sin(pitch)
        angle = float(np.rad2deg(np.arctan2(dir_y, dir_x)))

        attach_x = x0 + direction * scale * 0.02
        center = (
            float(attach_x + dir_x * length * 0.5),
            float(cy + dir_y * length * 0.5),
        )
        mask, t, v = leaf_geometry(h, w, center, length, leaf_width, angle)
        if not mask.any():
            continue

        # 最下面的几片老叶发黄：模拟缺氮 / 缺水 / 衰老导致的黄化
        if i < yellow_leaves:
            leaf_hue = float(np.clip(rng.uniform(21, 26), 0, 179))
            leaf_sat = float(rng.uniform(170, 200))
            leaf_val = float(rng.uniform(185, 210))
        else:
            leaf_hue = float(np.clip(hue + rng.normal(0, 4), 0, 179))
            leaf_sat = sat
            leaf_val = val + rng.normal(0, 8)

        paint_leaf(canvas, mask, t, v, rng, leaf_hue, leaf_sat, leaf_val, leaf_width)
        spot_total += add_spots(canvas, mask, rng, spots)

    return spot_total


def finish(canvas: np.ndarray, rng: np.random.Generator, blur: int = 3) -> np.ndarray:
    """收尾：轻微模糊 + 噪点 + 二次暗角，让画面更像相机拍摄。"""
    out = canvas.astype(np.float32)
    if blur > 1:
        out = cv2.GaussianBlur(out, (blur, blur), 0)
    out += rng.normal(0, 3.2, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return enable_alpha(out)


def enable_alpha(img: np.ndarray) -> np.ndarray:
    """转成 PNG 友好的 4 通道数组，方便直接写盘。"""
    alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.dstack([img, alpha])


def save_png(path: Path, bgra: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError(f"写入失败：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))


def sample_healthy_plant(seed: int) -> np.ndarray:
    """单株健康植物特写。"""
    rng = np.random.default_rng(seed)
    canvas = soil_background(rng, WIDTH, HEIGHT)
    draw_plant(canvas, rng, (WIDTH * 0.5, HEIGHT * 0.88), 300, 7)
    return finish(canvas, rng)


def sample_seedlings(seed: int) -> np.ndarray:
    """多株幼苗，用于测试计数功能。"""
    rng = np.random.default_rng(seed + 1)
    canvas = soil_background(rng, WIDTH, HEIGHT)
    bases = [(95, 465), (270, 500), (445, 455), (620, 495)]
    for bx, by in bases:
        draw_plant(
            canvas, rng, (bx, by), float(rng.uniform(105, 125)), 5,
            hue=float(rng.uniform(46, 58)), val=float(rng.uniform(135, 160)),
            pitch_range=(38.0, 62.0),
        )
    return finish(canvas, rng)


def sample_leaf_disease(seed: int) -> np.ndarray:
    """病害叶片：叶面散布褐色病斑，下部老叶明显发黄。"""
    rng = np.random.default_rng(seed + 2)
    canvas = soil_background(rng, WIDTH, HEIGHT)
    draw_plant(
        canvas, rng, (WIDTH * 0.5, HEIGHT * 0.92), 300, 7,
        hue=48.0, sat=150.0, val=145.0, spots=4, yellow_leaves=3,
    )
    return finish(canvas, rng, blur=3)


def sample_no_plant(seed: int) -> np.ndarray:
    """干扰样本：只有土壤、石子和一根枯枝，用来验证不漏检为植物。"""
    rng = np.random.default_rng(seed + 3)
    canvas = soil_background(rng, WIDTH, HEIGHT)
    for _ in range(40):
        cx, cy = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        r = int(rng.integers(4, 14))
        tone = float(rng.uniform(90, 165))
        cv2.circle(canvas, (cx, cy), r, (tone, tone * 0.95, tone * 0.88), -1)
    # 枯枝
    cv2.line(
        canvas, (180, 420), (560, 150), bgr_tuple(18, 90, 120), 9, cv2.LINE_AA
    )
    cv2.line(
        canvas, (380, 280), (470, 240), bgr_tuple(18, 90, 110), 6, cv2.LINE_AA
    )
    return finish(canvas, rng)


SAMPLES = {
    "01_healthy_plant.png": sample_healthy_plant,
    "02_seedlings.png": sample_seedlings,
    "03_leaf_disease.png": sample_leaf_disease,
    "04_no_plant.png": sample_no_plant,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成演示用合成植物图片")
    parser.add_argument("--out", default="samples", help="输出目录")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    args = parser.parse_args()

    out_dir = Path(args.out)
    for name, maker in SAMPLES.items():
        path = out_dir / name
        save_png(path, maker(args.seed))
        print(f"[生成] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
