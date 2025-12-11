"""图像处理服务测试模块"""

import sys
import time
from pathlib import Path

import cv2

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modes.imageTool import image_service  # noqa: E402


def test_detect_text_regions_morphology():
    """测试：形态学检测文字区域"""
    print("\n" + "=" * 60)
    print("测试 1: 形态学检测文字区域")
    print("=" * 60)

    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/3.detect/morphology_text_regions"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 执行形态学检测
        regions = image_service.detect_text_regions_morphology(
            image_path=image_path,
            min_area=100,
            max_area=10000,
            min_aspect_ratio=0.1,
            max_aspect_ratio=10.0,
        )

        elapsed_time = time.time() - start_time

        print(f"✅ 检测完成，共检测到 {len(regions)} 个文字区域")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")

        if regions:
            # 准备绘制数据
            boxes = [region["box"] for region in regions]

            # 绘制检测框（不显示标签）
            output_image_path = output_dir / "detection_result.png"
            image_service.draw_boxes(
                image_path=image_path,
                boxes=boxes,
                output_path=output_image_path,
                color=(0, 255, 0),  # 绿色
                thickness=2,
            )
            print(f"🖼️  检测结果已保存至: {output_image_path}")

            # 显示前10个检测结果
            print("\n前10个检测结果:")
            for idx, region in enumerate(regions[:10], 1):
                box = region["box"]
                center = region["center"]
                area = region["area"]
                print(f"  {idx:2d}. Box: {box} | Center: ({center[0]:.1f}, {center[1]:.1f}) | Area: {area}")

        else:
            print("⚠️  未检测到任何文字区域")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_detect_single_chars_morphology():
    """测试：形态学检测单字级区域"""
    print("\n" + "=" * 60)
    print("测试 2: 形态学检测单字级区域")
    print("=" * 60)

    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/3.detect/morphology_single_chars"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 执行单字级检测
        regions = image_service.detect_single_chars_morphology(
            image_path=image_path,
            char_size_range=(20, 200),
            min_area_ratio=0.3,
        )

        elapsed_time = time.time() - start_time

        print(f"✅ 检测完成，共检测到 {len(regions)} 个单字区域")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")

        if regions:
            # 准备绘制数据
            boxes = [region["box"] for region in regions]

            # 绘制检测框（不显示标签）
            output_image_path = output_dir / "detection_result.png"
            image_service.draw_boxes(
                image_path=image_path,
                boxes=boxes,
                output_path=output_image_path,
                color=(255, 0, 0),  # 红色
                thickness=2,
            )
            print(f"🖼️  检测结果已保存至: {output_image_path}")

            # 显示前10个检测结果
            print("\n前10个检测结果:")
            for idx, region in enumerate(regions[:10], 1):
                box = region["box"]
                center = region["center"]
                area = region["area"]
                print(f"  {idx:2d}. Box: {box} | Center: ({center[0]:.1f}, {center[1]:.1f}) | Area: {area}")

        else:
            print("⚠️  未检测到任何单字区域")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_detect_with_mser():
    """测试：MSER检测文字区域"""
    print("\n" + "=" * 60)
    print("测试 3: MSER检测文字区域")
    print("=" * 60)

    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/3.detect/mser"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 执行MSER检测
        regions = image_service.detect_with_mser(
            image_path=image_path,
            delta=5,
            min_area=100,
            max_area=14400,
            max_variation=0.25,
            min_size=10,
        )

        elapsed_time = time.time() - start_time

        print(f"✅ 检测完成，共检测到 {len(regions)} 个区域")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")

        if regions:
            # 准备绘制数据
            boxes = [region["box"] for region in regions]

            # 绘制检测框（不显示标签）
            output_image_path = output_dir / "detection_result.png"
            image_service.draw_boxes(
                image_path=image_path,
                boxes=boxes,
                output_path=output_image_path,
                color=(0, 0, 255),  # 蓝色
                thickness=2,
            )
            print(f"🖼️  检测结果已保存至: {output_image_path}")

            # 显示前10个检测结果
            print("\n前10个检测结果:")
            for idx, region in enumerate(regions[:10], 1):
                box = region["box"]
                center = region["center"]
                area = region["area"]
                print(f"  {idx:2d}. Box: {box} | Center: ({center[0]:.1f}, {center[1]:.1f}) | Area: {area}")

        else:
            print("⚠️  未检测到任何区域")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_combined_detection():
    """测试：组合多种检测方法"""
    print("\n" + "=" * 60)
    print("测试 4: 组合多种检测方法")
    print("=" * 60)

    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/3.detect/combined"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 执行多种检测方法
        morphology_regions = image_service.detect_text_regions_morphology(image_path)
        single_char_regions = image_service.detect_single_chars_morphology(image_path)
        mser_regions = image_service.detect_with_mser(image_path)

        elapsed_time = time.time() - start_time

        print("✅ 检测完成")
        print(f"  - 形态学检测: {len(morphology_regions)} 个区域")
        print(f"  - 单字级检测: {len(single_char_regions)} 个区域")
        print(f"  - MSER检测: {len(mser_regions)} 个区域")
        print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")

        # 使用不同颜色绘制三种检测方法的结果
        # 读取原始图像
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 创建组合图像（复制原始图像）
        combined_image = original_image.copy()

        # 绘制形态学检测结果（绿色）
        for region in morphology_regions:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制单字级检测结果（红色）
        for region in single_char_regions:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(combined_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制MSER检测结果（蓝色）
        for region in mser_regions:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 保存组合结果
        output_image_path = output_dir / "combined_result.png"
        cv2.imwrite(str(output_image_path), combined_image)
        print(f"🖼️  组合检测结果已保存至: {output_image_path}")
        print("  - 绿色框: 形态学检测")
        print("  - 红色框: 单字级检测")
        print("  - 蓝色框: MSER检测")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "🚀 " * 20)
    print("图像处理服务测试 - 形态学检测")
    print("🚀 " * 20)

    # 运行测试
    test_detect_text_regions_morphology()
    test_detect_single_chars_morphology()
    test_detect_with_mser()
    test_combined_detection()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    output_base = PROJECT_ROOT / "test_file/3.detect"
    print(f"\n📁 输出目录: {output_base}/")
    print("  - morphology_text_regions/ : 形态学检测文字区域结果")
    print("  - morphology_single_chars/ : 形态学检测单字级区域结果")
    print("  - mser/                   : MSER检测结果")
    print("  - combined/               : 组合检测结果")


if __name__ == "__main__":
    main()
