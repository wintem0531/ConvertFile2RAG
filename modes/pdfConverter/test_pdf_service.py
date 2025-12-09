"""PDF转图像服务测试模块"""

import sys
import time
from pathlib import Path

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modes.pdfConverter import pdf_service  # noqa: E402


def test_pdf_to_images():
    """测试：将PDF的所有页面转换为图像"""
    print("\n" + "=" * 60)
    print("测试 1: 将PDF的所有页面转换为图像")
    print("=" * 60)

    pdf_path = PROJECT_ROOT / "test_file/input/齊系文字編.pdf"
    output_dir = PROJECT_ROOT / "test_file/1.pdf2png/all_pages"

    if not pdf_path.exists():
        print(f"❌ 测试文件不存在: {pdf_path}")
        return

    print(f"📄 PDF文件: {pdf_path}")
    print(f"📁 输出目录: {output_dir}")

    start_time = time.time()
    try:
        image_paths = pdf_service.pdf_to_images(
            pdf_path=pdf_path,
            output_dir=output_dir,
            dpi=200,
            image_format="png",
            use_async=True,  # 使用异步并行处理
        )
        elapsed_time = time.time() - start_time

        print("✅ 转换成功！")
        print(f"📊 生成图像数量: {len(image_paths)}")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
        print(f"📈 平均每页: {elapsed_time / len(image_paths):.2f} 秒")

        # 显示前3个文件路径
        if image_paths:
            print("\n前3个生成的图像:")
            for i, path in enumerate(image_paths[:3], 1):
                print(f"  {i}. {Path(path).name}")

    except Exception as e:
        print(f"❌ 转换失败: {e}")


def test_pdf_page_to_image():
    """测试：将PDF的指定页转换为图像"""
    print("\n" + "=" * 60)
    print("测试 2: 将PDF的指定页转换为图像")
    print("=" * 60)

    pdf_path = PROJECT_ROOT / "test_file/input/齊系文字編.pdf"
    output_dir = PROJECT_ROOT / "test_file/1.pdf2png/single_page"
    page_number = 1  # 转换第1页

    if not pdf_path.exists():
        print(f"❌ 测试文件不存在: {pdf_path}")
        return

    print(f"📄 PDF文件: {pdf_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📑 页码: {page_number}")

    start_time = time.time()
    try:
        image_path = pdf_service.pdf_page_to_image(
            pdf_path=pdf_path,
            page_number=page_number,
            output_dir=output_dir,
            dpi=200,
            image_format="png",
        )
        elapsed_time = time.time() - start_time

        print("✅ 转换成功！")
        print(f"📊 生成图像: {Path(image_path).name}")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")

    except Exception as e:
        print(f"❌ 转换失败: {e}")


def test_pdf_pages_range_to_images():
    """测试：将PDF的指定页数范围转换为图像"""
    print("\n" + "=" * 60)
    print("测试 3: 将PDF的指定页数范围转换为图像")
    print("=" * 60)

    pdf_path = PROJECT_ROOT / "test_file/input/齊系文字編.pdf"
    output_dir = PROJECT_ROOT / "test_file/1.pdf2png/page_range"
    start_page = 1
    end_page = 5  # 转换第1-5页

    if not pdf_path.exists():
        print(f"❌ 测试文件不存在: {pdf_path}")
        return

    print(f"📄 PDF文件: {pdf_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📑 页码范围: {start_page} - {end_page}")

    start_time = time.time()
    try:
        image_paths = pdf_service.pdf_pages_range_to_images(
            pdf_path=pdf_path,
            start_page=start_page,
            end_page=end_page,
            output_dir=output_dir,
            dpi=200,
            image_format="png",
            use_async=True,  # 使用异步并行处理
        )
        elapsed_time = time.time() - start_time

        print("✅ 转换成功！")
        print(f"📊 生成图像数量: {len(image_paths)}")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
        print(f"📈 平均每页: {elapsed_time / len(image_paths):.2f} 秒")

        # 显示所有生成的文件
        print("\n生成的图像:")
        for i, path in enumerate(image_paths, 1):
            print(f"  {i}. {Path(path).name}")

    except Exception as e:
        print(f"❌ 转换失败: {e}")


def test_system_info():
    """显示系统配置信息"""
    print("\n" + "=" * 60)
    print("系统配置信息")
    print("=" * 60)

    import os

    cpu_count = os.cpu_count()
    optimal_workers = pdf_service.get_optimal_workers()

    print(f"🖥️  CPU核心数: {cpu_count}")
    print(f"⚙️  最优并发数: {optimal_workers}")


def main():
    """运行所有测试"""
    print("\n" + "🚀 " * 20)
    print("PDF转图像服务测试")
    print("🚀 " * 20)

    # 显示系统信息
    test_system_info()

    # 运行测试
    test_pdf_page_to_image()
    test_pdf_pages_range_to_images()
    test_pdf_to_images()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    print("\n📁 输出目录: test_file/1.pdf2png/")
    print("  - all_pages/     : 全部页面转换结果")
    print("  - single_page/   : 单页转换结果")
    print("  - page_range/    : 页码范围转换结果")


if __name__ == "__main__":
    main()
