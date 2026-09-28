"""plantdet - 一个不需要预训练模型的植物检测小工具包。

处理流程：

    图像
      -> 植被分割（超绿指数 ExG + HSV 颜色区间 + Otsu 自动阈值）
      -> 形态学清理（去噪点、补空洞）
      -> 连通域分析（植株计数、边界框、面积）
      -> 叶片状态评估（绿色占比、黄化占比、病斑数量、健康分）
      -> 可视化标注图 + JSON 报告

整个项目只依赖 numpy / opencv / matplotlib，拿到图片就能跑。
"""

from .analyze import AnalyzeConfig, Region, analyze_regions
from .pipeline import Result, analyze_image, analyze_path
from .segment import SegmentConfig, compute_exg, vegetation_mask

__version__ = "1.0.0"

__all__ = [
    "AnalyzeConfig",
    "Region",
    "Result",
    "SegmentConfig",
    "analyze_image",
    "analyze_path",
    "analyze_regions",
    "compute_exg",
    "vegetation_mask",
    "__version__",
]
