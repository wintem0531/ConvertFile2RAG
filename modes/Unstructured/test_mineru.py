"""MinerU 图片分析测试模块"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mineru.cli.common import do_parse, read_fn  # noqa: E402
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox  # noqa: E402


def visualize_mineru_results(image_path: Path, middle_json: dict, output_path: Path, pdf_bytes: bytes) -> None:
    """
    可视化 MinerU 分析结果

    Args:
        image_path: 原始图片路径
        middle_json: MinerU 返回的中间 JSON 结果
        output_path: 输出图片路径
        pdf_bytes: PDF 字节数据（用于绘制边界框）
    """
    # 读取原始图片
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 创建可视化图片（复制原始图片）
    vis_image = image.copy()

    # 定义不同元素类型的颜色
    color_map = {
        "title": (0, 255, 255),  # 黄色
        "text": (0, 255, 0),  # 绿色
        "list": (255, 0, 0),  # 蓝色
        "table": (255, 0, 255),  # 洋红色
        "figure": (0, 165, 255),  # 橙色
        "image": (128, 0, 128),  # 紫色
        "formula": (255, 255, 0),  # 青色
    }

    # 统计信息
    element_stats = {}

    # 从 middle_json 中提取页面信息
    # pdf_info 是一个数组，每个元素代表一页
    pdf_info = middle_json.get("pdf_info", [])

    for page_idx, page in enumerate(pdf_info):
        # 获取页面中的预处理块
        preproc_blocks = page.get("preproc_blocks", [])

        # 绘制每个块的边界框
        for block in preproc_blocks:
            block_type = block.get("type", "Unknown")
            bbox = block.get("bbox", [])

            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])

                # 确保坐标在图片范围内
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # 选择颜色
                color = color_map.get(block_type.lower(), (255, 255, 255))  # 默认白色

                # 绘制边界框（不绘制标签，避免遮挡）
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # 统计元素类型
                element_stats[block_type] = element_stats.get(block_type, 0) + 1

    # 保存可视化结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 元素类型统计")
    print("=" * 60)
    for elem_type, count in sorted(element_stats.items()):
        print(f"  {elem_type}: {count} 个")


def test_mineru_image_analysis():
    """测试 MinerU 图片分析"""
    print("\n" + "=" * 60)
    print("MinerU 图片分析测试")
    print("=" * 60)

    # 测试图片路径
    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_25.png"
    output_dir = PROJECT_ROOT / "test_file/5.mineru"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图片: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 使用 MinerU 分析图片
        print("\n🔍 开始分析图片...")
        print("  后端: vlm-transformers")
        print("  设备: 尝试使用 MPS (Apple Silicon Metal 加速)")
        print("  语言: 中文")

        # 将图片转换为 PDF 字节（MinerU 需要 PDF 格式）
        pdf_bytes = read_fn(image_path)
        pdf_file_name = image_path.stem

        # 尝试启用 MLX/MPS 加速
        # 注意：MinerU 的 transformers 后端使用 torch
        # 在 Apple Silicon 上，可以通过设置环境变量或使用 MPS 设备来加速
        # 虽然不能直接使用 MLX，但 MPS 可以提供类似的加速效果
        import os

        # 设置环境变量以尝试使用 MPS（如果可用）
        if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # 调用 do_parse 进行分析
        # 使用 vlm-transformers 后端，它会自动检测并使用 MPS（如果可用）
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend="vlm-mlx-engine",  # 使用 transformers 后端（支持 MPS 加速）
            parse_method="vlm",
            p_formula_enable=True,
            p_table_enable=True,
            f_draw_layout_bbox=True,
            f_draw_span_bbox=True,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=True,
            f_dump_orig_pdf=True,
            f_dump_content_list=True,
        )

        elapsed_time = time.time() - start_time

        print("✅ 分析完成")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")

        # 读取生成的中间 JSON 文件
        middle_json_path = output_dir / pdf_file_name / "vlm" / f"{pdf_file_name}_middle.json"

        if middle_json_path.exists():
            import json

            with open(middle_json_path, encoding="utf-8") as f:
                middle_json = json.load(f)

            # 可视化结果
            print("\n🎨 生成可视化结果...")
            vis_output_path = output_dir / "visualization_result.png"
            visualize_mineru_results(image_path, middle_json, vis_output_path, pdf_bytes)
            print(f"🖼️  可视化结果已保存至: {vis_output_path}")

            # 显示分析结果摘要
            pdf_info = middle_json.get("pdf_info", [])
            print(f"\n📄 共分析 {len(pdf_info)} 页")

            # 显示第一页的详细信息
            if pdf_info:
                first_page = pdf_info[0]
                preproc_blocks = first_page.get("preproc_blocks", [])
                para_blocks = first_page.get("para_blocks", [])

                print("\n📝 第一页详细信息:")
                print(f"  - 预处理块: {len(preproc_blocks)} 个")
                print(f"  - 段落块: {len(para_blocks)} 个")

                # 显示前5个预处理块
                print("\n前5个预处理块:")
                for idx, block in enumerate(preproc_blocks[:5], 1):
                    block_type = block.get("type", "Unknown")
                    bbox = block.get("bbox", [])
                    print(f"  {idx}. 类型: {block_type}, 位置: {bbox}")

        # 显示输出文件列表
        print("\n" + "=" * 60)
        print("📁 生成的文件")
        print("=" * 60)
        result_dir = output_dir / pdf_file_name / "vlm"
        if result_dir.exists():
            for file in sorted(result_dir.glob("*")):
                if file.is_file():
                    file_size = file.stat().st_size
                    print(f"  - {file.name} ({file_size / 1024:.1f} KB)")

        print("\n✅ 所有测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    """运行测试"""
    print("\n" + "🚀 " * 20)
    print("MinerU 图片分析测试")
    print("🚀 " * 20)

    test_mineru_image_analysis()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    output_base = PROJECT_ROOT / "test_file/5.mineru"
    print(f"\n📁 输出目录: {output_base}/")
    print("  - visualization_result.png: 可视化结果（带标注）")
    print("  - {图片名}/vlm/          : 详细分析结果")


if __name__ == "__main__":
    main()
