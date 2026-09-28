"""命令行入口：python -m plantdet.cli --input 图片或文件夹。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import AnalyzeConfig
from .pipeline import analyze_path
from .report import format_console
from .segment import SegmentConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plantdet",
        description="植物检测：植被分割 + 植株计数 + 叶片健康度评估（无需训练模型）",
    )
    parser.add_argument("-i", "--input", required=True, help="图片路径或图片文件夹")
    parser.add_argument("-o", "--out", default="results", help="结果输出目录")
    parser.add_argument("--hue-min", type=int, default=SegmentConfig.hue_min, help="绿色 H 下限")
    parser.add_argument("--hue-max", type=int, default=SegmentConfig.hue_max, help="绿色 H 上限")
    parser.add_argument("--sat-min", type=int, default=SegmentConfig.sat_min, help="饱和度下限")
    parser.add_argument("--val-min", type=int, default=SegmentConfig.val_min, help="亮度下限")
    parser.add_argument(
        "--yellow-connect",
        type=int,
        default=SegmentConfig.yellow_connect_px,
        help="黄叶允许离绿色主体的最大距离（像素），0 = 不把黄叶算作植物",
    )
    parser.add_argument("--exg-bias", type=float, default=0.0, help="ExG 阈值微调，正数更严格")
    parser.add_argument(
        "--min-area",
        type=float,
        default=SegmentConfig.min_area_ratio,
        help="碎块过滤阈值（相对整图面积）",
    )
    parser.add_argument("--no-figure", action="store_true", help="不生成三联对比图")
    parser.add_argument("--quiet", action="store_true", help="只输出一行汇总")
    return parser


def collect_images(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [target]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = Path(args.input)
    if not target.exists():
        print(f"[错误] 找不到输入：{target}")
        return 2

    images = collect_images(target)
    if not images:
        print(f"[错误] {target} 里没有找到图片")
        return 2

    segment_cfg = SegmentConfig(
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        sat_min=args.sat_min,
        val_min=args.val_min,
        yellow_connect_px=args.yellow_connect,
        exg_bias=args.exg_bias,
        min_area_ratio=args.min_area,
    )
    analyze_cfg = AnalyzeConfig()

    for image_path in images:
        result = analyze_path(
            image_path,
            out_dir=args.out,
            segment_cfg=segment_cfg,
            analyze_cfg=analyze_cfg,
            write_figure=not args.no_figure,
        )
        s = result.summary
        if args.quiet:
            print(
                f"[{result.name}] 覆盖度 {s['coverage_ratio'] * 100:.1f}% | "
                f"植株数 {s['estimated_plant_count']} | "
                f"健康分 {s['mean_health_score']}（{s['overall_label']}）| "
                f"病斑 {s['total_spot_count']} 处"
            )
        else:
            print(format_console(result))
        for key, value in result.outputs.items():
            print(f"  [{key}] {value}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
