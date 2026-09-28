"""一键演示：生成示例图片 -> 跑检测 -> 打印报告 -> 汇总成一张总览图。

用法：
    python run_demo.py

产出（默认都在 results/ 目录下）：
    *_detected.png   标注图：绿色=植物，红色=疑似病斑，方框=检测到的植株
    *_mask.png       植被掩膜
    *_report.json    结构化检测报告
    *_figure.png     原图 / 掩膜 / 结果 三联对比图
    summary_board.png  四张示例的总览图
    summary.csv        批量统计表
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plantdet.pipeline import analyze_path
from plantdet.report import format_console
from plantdet.visualize import _setup_chinese_font

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
RESULTS_DIR = ROOT / "results"

CASES = [
    ("01_healthy_plant.png", "单株健康植物：验证分割与健康度"),
    ("02_seedlings.png", "多株幼苗：验证植株计数"),
    ("03_leaf_disease.png", "病害植株：验证黄化与病斑检测"),
    ("04_no_plant.png", "无植物场景：验证不误检"),
]


def ensure_samples() -> None:
    if all((SAMPLES_DIR / name).exists() for name, _ in CASES):
        return
    print("示例图片不存在，先自动生成……")
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_demo_images  # noqa: PLC0415  延迟导入，避免不必要的依赖

    for name, maker in make_demo_images.SAMPLES.items():
        make_demo_images.save_png(SAMPLES_DIR / name, maker(7))
        print(f"[生成] {SAMPLES_DIR / name}")


def main() -> int:
    ensure_samples()
    _setup_chinese_font()

    rows = []
    board = []
    for name, description in CASES:
        path = SAMPLES_DIR / name
        result = analyze_path(path, out_dir=RESULTS_DIR)
        print(format_console(result))
        print(f"说明：{description}")
        for key, value in result.outputs.items():
            print(f"  [{key}] {value}")
        print()

        rows.append(
            {
                "图片": name,
                "覆盖度(%)": round(result.summary["coverage_ratio"] * 100, 1),
                "估计植株数": result.summary["estimated_plant_count"],
                "区域数": result.summary["region_count"],
                "健康分": result.summary["mean_health_score"],
                "总体判定": result.summary["overall_label"],
                "黄化占比(%)": round(result.summary["yellow_ratio"] * 100, 1),
                "疑似病斑数": result.summary["total_spot_count"],
            }
        )
        board.append((result, description))

    write_csv(rows, RESULTS_DIR / "summary.csv")
    make_board(board, RESULTS_DIR / "summary_board.png")
    print(f"批量统计表：{RESULTS_DIR / 'summary.csv'}")
    print(f"总览图：{RESULTS_DIR / 'summary_board.png'}")
    return 0


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_board(board, path: Path) -> Path:
    """把四张标注图拼成 2x2 总览图。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), dpi=140)
    for ax, (result, description) in zip(axes.ravel(), board):
        s = result.summary
        ax.imshow(cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{description}\n"
            f"覆盖度 {s['coverage_ratio'] * 100:.1f}% · 植株数 "
            f"{s['estimated_plant_count']} · 健康分 {s['mean_health_score']}"
            f"（{s['overall_label']}）· 病斑 {s['total_spot_count']} 处",
            fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
