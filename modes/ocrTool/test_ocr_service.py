"""OCR服务测试模块"""

import sys
import time
from pathlib import Path

import cv2

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modes.imageTool import image_service  # noqa: E402
from modes.ocrTool import ocr_service  # noqa: E402


def convert_box_to_rect(box: list) -> tuple[int, int, int, int]:
    """
    将OCR返回的box格式转换为矩形框格式

    Args:
        box: OCR返回的box格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        矩形框格式 (x_min, y_min, x_max, y_max)
    """
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))
    return (x_min, y_min, x_max, y_max)


def test_ocr_process_image():
    """测试完整的OCR处理流程（包含检测、识别、提取和框线标注）"""
    print("\n" + "=" * 60)
    print("OCR服务测试：完整处理流程")
    print("=" * 60)

    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/2.ocr"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    char_images_dir = output_dir / "char_images"
    char_images_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 使用完整处理流程（内部会调用 detect_characters 和 extract_character_images）
        print("\n🔍 开始OCR完整处理流程...")
        result = ocr_service.process_image(image_path, char_images_dir, save_character_images=True)

        elapsed_time = time.time() - start_time

        character_results = result["characters"]
        resized_image = result["resized_image"]
        scale = result["scale"]

        if not character_results:
            print("⚠️  未检测到任何字符")
            return

        print(f"✅ 处理完成，共检测到 {len(character_results)} 个字符")
        print(f"📏 图像缩放比例: {scale:.4f}")

        # 准备绘制数据
        print("\n🎨 准备绘制检测框...")
        boxes_to_draw = []

        for char_info in character_results:
            box = char_info["box"]

            # 转换box格式
            rect_box = convert_box_to_rect(box)
            boxes_to_draw.append(rect_box)

        print(f"✅ 准备完成，共 {len(boxes_to_draw)} 个检测框")

        # 保存缩放后的图像
        print("\n💾 保存缩放后的图像...")
        resized_image_path = output_dir / "resized_image.png"
        cv2.imwrite(str(resized_image_path), resized_image)
        print(f"✅ 缩放后的图像已保存至: {resized_image_path}")

        # 在缩放后的图像上绘制检测框（不带标签）
        print("\n🖼️  在缩放后的图像上绘制检测框...")
        output_image_path = output_dir / "detection_result.png"
        image_service.draw_boxes(
            image_path=resized_image_path,
            boxes=boxes_to_draw,
            output_path=output_image_path,
            color=(0, 255, 0),  # 绿色
            thickness=2,
        )
        print(f"✅ 绘制完成，结果保存至: {output_image_path}")

        # 统计信息
        avg_confidence = sum(char["confidence"] for char in character_results) / len(character_results)

        print("\n" + "=" * 60)
        print("📊 处理结果统计")
        print("=" * 60)
        print(f"📈 检测字符数量: {len(character_results)}")
        print(f"📈 平均置信度: {avg_confidence:.4f}")
        print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
        print(f"📏 图像缩放比例: {scale:.4f}")
        print(f"📁 字符图像目录: {char_images_dir}")
        print(f"🖼️  缩放后的图像: {resized_image_path}")
        print(f"🖼️  检测结果图像: {output_image_path}")

        # 显示前10个检测结果
        print("\n" + "=" * 60)
        print("📝 前10个检测结果示例")
        print("=" * 60)
        for idx, char_info in enumerate(character_results, 1):
            text = char_info["text"]
            confidence = char_info["confidence"]
            box = char_info["box"]
            rect_box = convert_box_to_rect(box)
            print(f"{idx:2d}. 文本: '{text}' | 置信度: {confidence:.4f} | 位置: {rect_box}")

        print("\n✅ 完整流程测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_ocr_detect_only():
    """测试只进行检测（det），不进行分类和识别，并绘制检测框"""
    print("\n" + "=" * 60)
    print("OCR服务测试：只检测（det）模式")
    print("=" * 60)

    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/2.ocr/detect_only"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 只进行检测（det），不进行分类和识别
        print("\n🔍 开始只检测模式（det only）...")
        boxes, resized_image, scale = ocr_service.detect_only(image_path)

        elapsed_time = time.time() - start_time

        print(f"✅ 检测完成，共检测到 {len(boxes)} 个文本框")
        print(f"📏 图像缩放比例: {scale:.4f}")

        if not boxes:
            print("⚠️  未检测到任何文本框")
            return

        # 准备绘制数据（将box转换为矩形框格式）
        print("\n🎨 准备绘制检测框...")
        boxes_to_draw = []

        for box in boxes:
            # 转换box格式：从 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 转换为 (x_min, y_min, x_max, y_max)
            rect_box = convert_box_to_rect(box)
            boxes_to_draw.append(rect_box)

        print(f"✅ 准备完成，共 {len(boxes_to_draw)} 个检测框")

        # 保存缩放后的图像
        print("\n💾 保存缩放后的图像...")
        resized_image_path = output_dir / "resized_image.png"
        cv2.imwrite(str(resized_image_path), resized_image)
        print(f"✅ 缩放后的图像已保存至: {resized_image_path}")

        # 在缩放后的图像上绘制检测框
        print("\n🖼️  在缩放后的图像上绘制检测框...")
        output_image_path = output_dir / "detect_only_result.png"
        image_service.draw_boxes(
            image_path=resized_image_path,
            boxes=boxes_to_draw,
            output_path=output_image_path,
            color=(255, 0, 0),  # 红色
            thickness=2,
        )
        print(f"✅ 绘制完成，结果保存至: {output_image_path}")

        # 统计信息
        print("\n" + "=" * 60)
        print("📊 检测结果统计")
        print("=" * 60)
        print(f"📈 检测文本框数量: {len(boxes)}")
        print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
        print(f"📏 图像缩放比例: {scale:.4f}")
        print(f"🖼️  缩放后的图像: {resized_image_path}")
        print(f"🖼️  检测结果图像: {output_image_path}")

        # 显示前10个检测框的位置
        print("\n" + "=" * 60)
        print("📝 前10个检测框位置示例")
        print("=" * 60)
        for idx, box in enumerate(boxes[:10], 1):
            rect_box = convert_box_to_rect(box)
            print(f"{idx:2d}. 位置: {rect_box}")

        print("\n✅ 只检测模式测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "🚀 " * 20)
    print("OCR服务测试")
    print("🚀 " * 20)

    # 测试1: 完整处理流程（包含检测、识别、提取和框线标注）
    test_ocr_process_image()

    # 测试2: 只检测模式（det only）
    test_ocr_detect_only()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {PROJECT_ROOT / 'test_file/2.ocr'}/")
    print("  - char_images/        : 提取的字符图像")
    print("  - resized_image.png   : 缩放后的原始图像")
    print("  - detection_result.png: 检测结果图像（带框线标注）")
    print("  - detect_only/        : 只检测模式结果")


if __name__ == "__main__":
    main()
