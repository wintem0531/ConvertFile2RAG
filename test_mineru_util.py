#!/usr/bin/env python3
"""测试 MinerU 工具函数"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入 MinerU 工具函数 - 必须在设置路径之后
from modes.mineru.mineru_util import (  # noqa: E402
    extract_content_list_from_pdf,
    extract_images_from_content_list,
    extract_text_blocks_from_content_list,
    extract_text_from_pdf,
)


def main():
    """主函数"""
    test_mineru_util()


def test_mineru_util():
    """测试 MinerU 工具函数"""
    print("=" * 60)
    print("MinerU 工具函数测试")
    print("=" * 60)

    # 测试图像路径 - 将被转换为 PDF 进行处理
    test_image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_25.png"
    output_dir = PROJECT_ROOT / "test_file/5.mineru"

    if not test_image_path.exists():
        print(f"❌ 测试文件不存在: {test_image_path}")
        return

    print(f"📄 测试文件: {test_image_path}")
    print(f"📁 输出目录: {output_dir}")

    try:
        # 测试提取内容列表
        print("\n🔍 测试提取内容列表...")
        content_list = extract_content_list_from_pdf(
            pdf_path=test_image_path,
            page_range=None,  # 全部页面
            output_dir=output_dir,
            save_result=True,
        )
        print(content_list)
        print(f"✅ 成功提取内容列表，共 {len(content_list)} 个元素")

        # 显示前5个内容元素
        print("\n前5个内容元素:")
        for idx, item in enumerate(content_list[:5], 1):
            item_type = item.get("type", "Unknown")
            text = item.get("text", "")[:50] + "..." if len(item.get("text", "")) > 50 else item.get("text", "")
            print(f"  {idx}. 类型: {item_type}, 文本: {text}")

        # 测试提取文本块
        print("\n🔍 测试提取文本块...")
        text_blocks = extract_text_blocks_from_content_list(content_list)
        print(f"✅ 成功提取文本块，共 {len(text_blocks)} 个")

        # 显示前3个文本块
        print("\n前3个文本块:")
        for idx, block in enumerate(text_blocks[:3], 1):
            text = block.get("text", "")[:100] + "..." if len(block.get("text", "")) > 100 else block.get("text", "")
            print(f"  {idx}. {text}")

        # 测试提取图片信息
        print("\n🔍 测试提取图片信息...")
        images = extract_images_from_content_list(content_list, output_dir)
        print(f"✅ 成功提取图片信息，共 {len(images)} 个")

        # 显示图片信息
        print("\n图片信息:")
        for idx, img in enumerate(images, 1):
            img_path = img.get("img_path", "")
            captions = img.get("image_caption", [])
            print(f"  {idx}. 路径: {img_path}, 说明: {captions}")

        # 测试直接提取文本
        print("\n🔍 测试直接提取文本...")
        text_content = extract_text_from_pdf(
            pdf_path=test_image_path,
            page_range=(1, 1),  # 只处理第一页
            output_dir=output_dir,
        )

        print(f"✅ 成功提取文本，共 {len(text_content)} 个字符")
        print(f"\n文本内容预览:\n{text_content[:200]}...")

        # 保存结果
        result_path = output_dir / f"{test_image_path.stem}_extracted_results.json"
        results = {
            "content_list": content_list,
            "text_blocks": text_blocks,
            "images": images,
            "text_content": text_content,
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 结果已保存至: {result_path}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
