# -*- coding: utf-8 -*-
"""MinerU 工具函数测试模块

测试 MinerU 解析功能，包括新的标准化输出接口。
"""

import json
import sys
import time
from pathlib import Path

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modes.mineru.mineru_util import (
    parse_pdf,
    parse_pdf_simple,
    extract_text_from_pdf,
    MinerUOutput,
    TextBlock,
    ImageBlock,
    TableBlock,
    EquationBlock,
)


def test_standardized_output_with_qi_system():
    """使用齊系文字編.pdf 测试新的标准化输出接口

    测试页面范围：1-10页和24-26页
    """
    print("="*80)
    print("测试 MinerU 标准化输出接口")
    print("="*80)

    # 测试文件路径
    test_pdf = PROJECT_ROOT / "test_file" / "input" / "齊系文字編.pdf"

    if not test_pdf.exists():
        print(f"❌ 测试文件不存在: {test_pdf}")
        print("请确保测试文件位于正确位置")
        return

    print(f"📄 测试文件: {test_pdf}")
    print(f"📊 文件大小: {test_pdf.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # 测试两个页面范围
    test_ranges = [
        {"pages": (1, 10), "name": "前10页"},
        {"pages": (24, 26), "name": "24-26页"},
    ]

    for backend in ["vlm-mlx-engine", "pipeline"]:
        print(f"\n{'='*60}")
        print(f"🔧 使用后端: {backend}")
        print(f"{'='*60}")

        for range_info in test_ranges:
            print(f"\n📖 解析 {range_info['name']} (页面 {range_info['pages'][0]}-{range_info['pages'][1]})")
            print("-" * 60)

            try:
                # 记录开始时间
                start_time = time.time()

                # 使用新的标准化接口解析
                output = parse_pdf(
                    pdf_path=test_pdf,
                    page_range=range_info["pages"],
                    backend=backend,
                    lang="ch",
                    formula_enable=True,
                    table_enable=True,
                    return_format="content_list"
                )

                # 记录结束时间
                end_time = time.time()
                elapsed = end_time - start_time

                # 验证输出类型
                assert isinstance(output, MinerUOutput), "输出应为 MinerUOutput 类型"

                # 统计信息
                print(f"✅ 解析成功！耗时: {elapsed:.2f} 秒")
                print(f"📋 总块数: {len(output.blocks)}")
                print(f"📝 文本块: {len(output.text_blocks)}")
                print(f"🖼️  图片块: {len(output.image_blocks)}")
                print(f"📊 表格块: {len(output.table_blocks)}")
                print(f"🔢 公式块: {len(output.equation_blocks)}")

                # 显示前几个文本块
                if output.text_blocks:
                    print("\n📌 前5个文本块示例:")
                    for i, block in enumerate(output.text_blocks[:5]):
                        preview = block.text[:100] + "..." if len(block.text) > 100 else block.text
                        print(f"  [{i+1}] L{block.text_level} | P{block.page_idx + 1}: {preview}")

                # 显示图片信息
                if output.image_blocks:
                    print(f"\n🖼️  图片块信息 (共 {len(output.image_blocks)} 个):")
                    for i, block in enumerate(output.image_blocks[:3]):
                        print(f"  [{i+1}] P{block.page_idx + 1}: {block.img_path}")
                        if block.caption:
                            print(f"      说明: {'; '.join(block.caption[:2])}")

                # 检查特定页面的内容
                if range_info["pages"][0] == 24:
                    print(f"\n📄 第25页内容预览:")
                    page_25_blocks = output.get_blocks_by_page(24)  # 第25页，索引为24
                    text_count = sum(1 for b in page_25_blocks if isinstance(b, TextBlock))
                    print(f"  - 文本块: {text_count} 个")
                    if text_count > 0:
                        first_text = next(b for b in page_25_blocks if isinstance(b, TextBlock))
                        preview = first_text.text[:150] + "..." if len(first_text.text) > 150 else first_text.text
                        print(f"  - 首个文本: {preview}")

                # 保存部分文本结果
                if output.plain_text:
                    # 创建输出目录
                    output_dir = PROJECT_ROOT / "test_file" / "output"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # 保存文本
                    text_file = output_dir / f"齊系文字編_{range_info['name'].replace('-', '_')}_{backend}.txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(f"文件: 齊系文字編.pdf\n")
                        f.write(f"页面: {range_info['name']}\n")
                        f.write(f"后端: {backend}\n")
                        f.write(f"解析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("="*60 + "\n\n")
                        f.write(output.plain_text)

                    print(f"\n💾 文本已保存到: {text_file.relative_to(PROJECT_ROOT)}")

                print("\n" + "✨" * 30 + " 成功 " + "✨" * 30)

            except Exception as e:
                print(f"❌ 解析失败: {str(e)}")
                import traceback
                print(f"错误详情:\n{traceback.format_exc()}")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "="*80)
    print("测试向后兼容性")
    print("="*80)

    test_pdf = PROJECT_ROOT / "test_file" / "input" / "齊系文字編.pdf"

    if not test_pdf.exists():
        print(f"❌ 测试文件不存在")
        return

    try:
        print("\n🔄 测试旧接口 extract_text_from_pdf()...")
        start_time = time.time()

        # 只解析第一页以加快速度
        text = extract_text_from_pdf(
            test_pdf,
            page_range=(1, 1),
            backend="vlm-mlx-engine"
        )

        elapsed = time.time() - start_time

        print(f"✅ 成功提取文本！耗时: {elapsed:.2f} 秒")
        print(f"📝 文本长度: {len(text)} 字符")
        print(f"📄 前200字符预览:\n{text[:200]}...")

    except Exception as e:
        print(f"❌ 旧接口测试失败: {e}")


def test_content_comparison():
    """比较不同后端的输出一致性"""
    print("\n" + "="*80)
    print("比较不同后端输出一致性")
    print("="*80)

    test_pdf = PROJECT_ROOT / "test_file" / "input" / "齊系文字編.pdf"

    if not test_pdf.exists():
        print(f"❌ 测试文件不存在")
        return

    # 只解析第25页进行比较
    page_range = (25, 25)
    outputs = {}

    for backend in ["vlm-mlx-engine", "pipeline"]:
        print(f"\n🔧 解析页面 {page_range[0]} 使用 {backend} 后端...")

        try:
            output = parse_pdf_simple(
                test_pdf,
                page_range=page_range,
                backend=backend
            )
            outputs[backend] = output

            print(f"  ✅ 文本块数: {len(output.text_blocks)}")
            if output.text_blocks:
                first_text = output.text_blocks[0].text[:100]
                print(f"  📝 首个文本块开头: {first_text}...")

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    # 比较结果
    if len(outputs) == 2:
        print("\n📊 输出比较:")
        backend1, backend2 = list(outputs.keys())
        output1, output2 = outputs[backend1], outputs[backend2]

        print(f"  {backend1} 文本块数: {len(output1.text_blocks)}")
        print(f"  {backend2} 文本块数: {len(output2.text_blocks)}")

        if output1.text_blocks and output2.text_blocks:
            text1 = output1.text_blocks[0].text[:200]
            text2 = output2.text_blocks[0].text[:200]

            similarity = 0
            if text1 and text2:
                # 简单的相似度检查
                common_chars = sum(c1 == c2 for c1, c2 in zip(text1, text2))
                similarity = common_chars / max(len(text1), len(text2)) * 100

            print(f"  首文本块相似度: {similarity:.1f}%")


def main():
    """主测试函数"""
    print("\n" + "🚀" * 40)
    print("MinerU 标准化接口测试 - 齊系文字編.pdf")
    print("🚀" * 40)

    # 运行主要测试
    test_standardized_output_with_qi_system()

    # 测试向后兼容性
    test_backward_compatibility()

    # 比较输出一致性
    test_content_comparison()

    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)

    # 输出使用说明
    print("\n📚 新接口使用示例:")
    print("""
from modes.mineru.mineru_util import parse_pdf_simple

# 基本使用
output = parse_pdf_simple("document.pdf")

# 访问文本
for text_block in output.text_blocks:
    print(f"级别: {text_block.text_level}, 文本: {text_block.text}")

# 获取纯文本
text = output.plain_text

# 获取 Markdown
md = output.markdown

# 按页面获取内容
page_content = output.get_blocks_by_page(0)
    """)


if __name__ == "__main__":
    main()