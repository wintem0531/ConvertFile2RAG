"""Unstructured 图片分析测试模块"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unstructured.partition.auto import partition  # noqa: E402


def visualize_unstructured_results(image_path: Path, elements: list, output_path: Path) -> None:
    """
    可视化 Unstructured 分析结果

    Args:
        image_path: 原始图片路径
        elements: Unstructured 返回的元素列表
        output_path: 输出图片路径
    """
    # 读取原始图片
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 创建可视化图片（复制原始图片）
    vis_image = image.copy()

    # 定义不同元素类型的颜色
    color_map = {
        "Title": (0, 255, 255),  # 黄色
        "NarrativeText": (0, 255, 0),  # 绿色
        "ListItem": (255, 0, 0),  # 蓝色
        "Table": (255, 0, 255),  # 洋红色
        "Figure": (0, 165, 255),  # 橙色
        "Image": (128, 0, 128),  # 紫色
        "PageBreak": (128, 128, 128),  # 灰色
    }

    # 统计信息
    element_stats = {}

    # 绘制每个元素的边界框
    for idx, element in enumerate(elements):
        element_type = element.category if hasattr(element, "category") else "Unknown"
        element_text = element.text if hasattr(element, "text") else ""

        # 获取元素的位置信息
        if hasattr(element, "metadata") and element.metadata:
            metadata = element.metadata
            # 尝试获取坐标信息
            coordinates = None
            if hasattr(metadata, "coordinates"):
                coordinates = metadata.coordinates
            elif hasattr(metadata, "bbox"):
                coordinates = metadata.bbox

            if coordinates:
                # 提取坐标
                if hasattr(coordinates, "x1"):
                    x1 = int(coordinates.x1)
                    y1 = int(coordinates.y1)
                    x2 = int(coordinates.x2)
                    y2 = int(coordinates.y2)
                elif isinstance(coordinates, (list, tuple)) and len(coordinates) >= 4:
                    x1, y1, x2, y2 = map(int, coordinates[:4])
                else:
                    # 如果没有坐标信息，跳过绘制
                    continue

                # 确保坐标在图片范围内
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # 选择颜色
                color = color_map.get(element_type, (255, 255, 255))  # 默认白色

                # 绘制边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"{element_type}:{idx}"
                if element_text:
                    # 截断文本，避免标签过长
                    text_preview = element_text[:20] + "..." if len(element_text) > 20 else element_text
                    label = f"{element_type}\n{text_preview}"

                # 计算文本大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label.split("\n")[0], font, font_scale, thickness)

                # 在框上方绘制文本背景
                text_y = max(y1 - 5, text_height + 5)
                cv2.rectangle(
                    vis_image,
                    (x1, text_y - text_height - 5),
                    (x1 + text_width + 10, text_y + baseline),
                    color,
                    -1,
                )

                # 绘制文本（多行）
                y_offset = text_y
                for line in label.split("\n"):
                    cv2.putText(
                        vis_image,
                        line,
                        (x1 + 5, y_offset),
                        font,
                        font_scale,
                        (255, 255, 255),  # 白色文本
                        thickness,
                        cv2.LINE_AA,
                    )
                    y_offset += text_height + 5

        # 统计元素类型
        element_stats[element_type] = element_stats.get(element_type, 0) + 1

    # 保存可视化结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 元素类型统计")
    print("=" * 60)
    for elem_type, count in sorted(element_stats.items()):
        print(f"  {elem_type}: {count} 个")


def test_unstructured_image_analysis():
    """测试 Unstructured 图片分析"""
    print("\n" + "=" * 60)
    print("Unstructured 图片分析测试")
    print("=" * 60)

    # 测试图片路径
    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/4.unstructured"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图片: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 使用 Unstructured 分析图片
        print("\n🔍 开始分析图片...")
        print("  策略: hi_res (高分辨率)")
        print("  OCR语言: 中文")

        elements = partition(
            filename=str(image_path),
            strategy="hi_res",
            languages=["chi_sim", "eng"],  # 中文简体和英文
            infer_table_structure=True,  # 推断表格结构
        )

        elapsed_time = time.time() - start_time

        print(f"✅ 分析完成，共检测到 {len(elements)} 个元素")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")

        # 保存分析结果到文本文件
        result_text_path = output_dir / "analysis_result.txt"
        with open(result_text_path, "w", encoding="utf-8") as f:
            f.write("Unstructured 分析结果\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"图片路径: {image_path}\n")
            f.write(f"检测到 {len(elements)} 个元素\n")
            f.write(f"分析耗时: {elapsed_time:.2f} 秒\n")
            f.write(f"{'=' * 60}\n\n")

            for idx, element in enumerate(elements, 1):
                element_type = element.category if hasattr(element, "category") else "Unknown"
                element_text = element.text if hasattr(element, "text") else ""

                f.write(f"\n元素 {idx}: {element_type}\n")
                f.write(f"{'-' * 40}\n")
                f.write(f"文本内容:\n{element_text}\n")

                # 写入元数据
                if hasattr(element, "metadata") and element.metadata:
                    metadata = element.metadata
                    f.write("\n元数据:\n")
                    if hasattr(metadata, "coordinates"):
                        f.write(f"  坐标: {metadata.coordinates}\n")
                    if hasattr(metadata, "page_number"):
                        f.write(f"  页码: {metadata.page_number}\n")

                f.write("\n")

        print(f"📝 分析结果已保存至: {result_text_path}")

        # 可视化结果
        print("\n🎨 生成可视化结果...")
        vis_output_path = output_dir / "visualization_result.png"
        visualize_unstructured_results(image_path, elements, vis_output_path)
        print(f"🖼️  可视化结果已保存至: {vis_output_path}")

        # 显示前10个元素的详细信息
        print("\n" + "=" * 60)
        print("📝 前10个元素详细信息")
        print("=" * 60)
        for idx, element in enumerate(elements[:10], 1):
            element_type = element.category if hasattr(element, "category") else "Unknown"
            element_text = element.text if hasattr(element, "text") else ""

            print(f"\n元素 {idx}: {element_type}")
            print(f"  文本: {element_text[:100]}{'...' if len(element_text) > 100 else ''}")

            # 显示坐标信息
            if hasattr(element, "metadata") and element.metadata:
                metadata = element.metadata
                if hasattr(metadata, "coordinates"):
                    coords = metadata.coordinates
                    if hasattr(coords, "x1"):
                        print(f"  位置: ({coords.x1}, {coords.y1}) - ({coords.x2}, {coords.y2})")

        print("\n✅ 所有测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    """运行测试"""
    print("\n" + "🚀 " * 20)
    print("Unstructured 图片分析测试")
    print("🚀 " * 20)

    test_unstructured_image_analysis()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    output_base = PROJECT_ROOT / "test_file/4.unstructured"
    print(f"\n📁 输出目录: {output_base}/")
    print("  - analysis_result.txt    : 详细分析结果")
    print("  - visualization_result.png: 可视化结果（带标注）")


if __name__ == "__main__":
    main()
