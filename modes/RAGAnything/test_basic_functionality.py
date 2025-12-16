#!/usr/bin/env python3
"""
RAG-Anything 基本功能测试

该脚本测试 RAG-Anything 的基本功能，包括文档解析和内容处理。
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class RAGAnythingBasicTester:
    """RAG-Anything 基本功能测试类"""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.test_files_dir = self.test_dir / "test_files"
        self.output_dir = self.test_dir / "output"

        # 确保目录存在
        self.test_files_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        print(f"测试文件目录: {self.test_files_dir}")
        print(f"输出目录: {self.output_dir}")

    async def test_raganything_import(self):
        """测试 RAG-Anything 导入"""
        print("\n测试 RAG-Anything 导入...")
        try:
            from raganything import RAGAnything, RAGAnythingConfig

            print("✅ RAG-Anything 导入成功")
            return RAGAnything, RAGAnythingConfig
        except ImportError as e:
            print(f"❌ RAG-Anything 导入失败: {str(e)}")
            print("请先运行 install_dependencies.py 安装依赖")
            return None, None

    async def test_mineru_availability(self):
        """测试 MinerU 可用性"""
        print("\n测试 MinerU 可用性...")
        try:
            import subprocess

            result = subprocess.run(["mineru", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ MinerU 可用: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ MinerU 不可用: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ MinerU 测试失败: {str(e)}")
            return False

    async def test_sample_document_processing(self, RAGAnything, RAGAnythingConfig):
        """测试示例文档处理"""
        print("\n测试示例文档处理...")

        # 创建一个简单的测试文档
        test_doc_path = self.create_sample_document()
        if not test_doc_path:
            print("❌ 无法创建测试文档")
            return False

        try:
            # 创建配置
            config = RAGAnythingConfig(
                working_dir=str(self.output_dir / "rag_storage"),
                parser="mineru",
                parse_method="auto",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )

            # 注意：在实际环境中，需要设置 API 密钥和嵌入函数
            # 这里仅测试 RAGAnything 的基本初始化和配置
            print("注意: 由于没有设置 API 密钥和嵌入函数，此测试仅验证配置和初始化")

            # 验证配置
            print("配置信息:")
            print(f"  工作目录: {config.working_dir}")
            print(f"  解析器: {config.parser}")
            print(f"  解析方法: {config.parse_method}")
            print(f"  图像处理: {config.enable_image_processing}")
            print(f"  表格处理: {config.enable_table_processing}")
            print(f"  公式处理: {config.enable_equation_processing}")

            print("✅ 配置创建成功")
            return True

        except Exception as e:
            print(f"❌ 文档处理测试失败: {str(e)}")
            return False

    def create_sample_document(self):
        """创建一个示例文档用于测试"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas

            # 创建 PDF 文档路径
            doc_path = self.test_files_dir / "test_document.pdf"

            # 创建 PDF
            c = canvas.Canvas(str(doc_path), pagesize=letter)

            # 添加标题
            c.setFont("Helvetica-Bold", 16)
            c.drawString(inch, 10 * inch, "RAG-Anything 测试文档")

            # 添加段落
            c.setFont("Helvetica", 12)
            text_lines = [
                "RAG-Anything 是一个综合性的多模态文档处理 RAG 系统。",
                "该系统能够处理包含文本、图像、表格和公式等多模态内容的复杂文档。",
                "它提供完整的检索增强(RAG)生成解决方案。",
            ]

            y_position = 9 * inch
            for line in text_lines:
                c.drawString(inch, y_position, line)
                y_position -= 0.3 * inch

            # 添加表格标题
            c.setFont("Helvetica-Bold", 12)
            c.drawString(inch, y_position - 0.2 * inch, "性能对比表:")
            y_position -= 0.5 * inch

            # 添加简单表格
            c.setFont("Helvetica", 10)
            table_data = [
                ("方法", "准确率", "F1分数"),
                ("RAG-Anything", "95.2%", "0.94"),
                ("基准方法", "87.3%", "0.85"),
            ]

            # 绘制表格
            for i, row in enumerate(table_data):
                x_pos = inch
                for cell in row:
                    c.drawString(x_pos, y_position - i * 0.2 * inch, cell)
                    x_pos += 1.5 * inch

            # 添加数学公式
            y_position -= len(table_data) * 0.2 * inch + 0.3 * inch
            c.setFont("Helvetica", 10)
            c.drawString(inch, y_position, "相关性公式: P(d|q) = P(q|d) * P(d) / P(q)")

            # 完成第一页
            c.showPage()

            # 添加第二页
            c.setFont("Helvetica-Bold", 16)
            c.drawString(inch, 10 * inch, "图像处理示例")

            c.setFont("Helvetica", 12)
            c.drawString(inch, 9 * inch, "本页用于测试图像处理功能。")
            c.drawString(inch, 8.5 * inch, "在实际应用中，RAG-Anything 能够识别和分析文档中的图像。")

            c.save()

            print(f"✅ 示例文档已创建: {doc_path}")
            return doc_path

        except ImportError:
            print("❌ reportlab 库未安装，无法创建 PDF 文档")
            print("请运行: uv install reportlab")
            return None
        except Exception as e:
            print(f"❌ 创建示例文档失败: {str(e)}")
            return None

    async def test_direct_parsing(self):
        """测试直接使用 MinerU 解析文档"""
        print("\n测试直接使用 MinerU 解析文档...")

        test_doc_path = self.create_sample_document()
        if not test_doc_path:
            print("❌ 无法创建测试文档")
            return False

        try:
            import subprocess

            output_dir = self.output_dir / "mineru_output"
            output_dir.mkdir(exist_ok=True)

            # 使用 MinerU 解析文档
            command = f"mineru -p {test_doc_path} -o {output_dir} -m auto"
            print(f"执行命令: {command}")

            start_time = time.time()
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"解析耗时: {execution_time:.2f} 秒")

            if result.returncode == 0:
                print("✅ MinerU 解析成功")
                print(f"输出目录: {output_dir}")

                # 列出输出文件
                output_files = list(output_dir.rglob("*"))
                print(f"输出文件数量: {len(output_files)}")
                for file in output_files:
                    if file.is_file():
                        print(f"  - {file.relative_to(output_dir)}")

                return True
            else:
                print(f"❌ MinerU 解析失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ 直接解析测试失败: {str(e)}")
            return False

    async def run_all_tests(self):
        """运行所有测试"""
        print("开始 RAG-Anything 基本功能测试")
        print("=" * 60)

        test_results = []

        # 测试 1: 导入测试
        RAGAnything, RAGAnythingConfig = await self.test_raganything_import()
        test_results.append(("RAG-Anything 导入", RAGAnything is not None))

        # 测试 2: MinerU 可用性
        mineru_available = await self.test_mineru_availability()
        test_results.append(("MinerU 可用性", mineru_available))

        # 测试 3: 示例文档处理
        if RAGAnything is not None:
            doc_processing_result = await self.test_sample_document_processing(RAGAnything, RAGAnythingConfig)
            test_results.append(("示例文档处理配置", doc_processing_result))

        # 测试 4: 直接解析
        direct_parse_result = await self.test_direct_parsing()
        test_results.append(("MinerU 直接解析", direct_parse_result))

        # 输出测试结果汇总
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)

        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")

        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)

        print(f"\n通过: {passed_tests}/{total_tests}")

        if passed_tests == total_tests:
            print("🎉 所有测试通过!")
        else:
            print("⚠️ 部分测试失败，请检查上述错误信息")

        # 保存测试结果
        results_file = self.test_dir / "test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "results": [{"name": name, "passed": result} for name, result in test_results],
                    "summary": {"total": total_tests, "passed": passed_tests, "failed": total_tests - passed_tests},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n测试结果已保存至: {results_file}")


async def main():
    """主函数"""
    tester = RAGAnythingBasicTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
