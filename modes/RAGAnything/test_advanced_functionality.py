#!/usr/bin/env python3
"""
RAG-Anything 高级功能测试

该脚本测试 RAG-Anything 的高级功能，包括完整的 RAG 流程和多模态内容处理。
"""

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class RAGAnythingAdvancedTester:
    """RAG-Anything 高级功能测试类"""

    def __init__(self, api_key=None, base_url=None):
        self.test_dir = Path(__file__).parent
        self.test_files_dir = self.test_dir / "test_files"
        self.output_dir = self.test_dir / "output"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-test-key-placeholder")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # 确保目录存在
        self.test_files_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        print(f"测试文件目录: {self.test_files_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"API Base URL: {self.base_url}")
        print(f"API Key: {'已设置' if self.api_key else '未设置'}")

    async def test_raganything_initialization(self):
        """测试 RAG-Anything 初始化"""
        print("\n测试 RAG-Anything 初始化...")
        try:
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc
            from raganything import RAGAnything, RAGAnythingConfig

            # 创建配置
            config = RAGAnythingConfig(
                working_dir=str(self.output_dir / "rag_storage"),
                parser="mineru",
                parse_method="auto",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )

            # 定义 LLM 模型函数
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    "gpt-4o-mini",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )

            # 定义视觉模型函数用于图像处理
            def vision_model_func(
                prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
            ):
                # 如果提供了messages格式（用于多模态VLM增强查询），直接使用
                if messages:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=messages,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        **kwargs,
                    )
                # 传统单图片格式
                elif image_data:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=[
                            {"role": "system", "content": system_prompt} if system_prompt else None,
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt},
                        ],
                        api_key=self.api_key,
                        base_url=self.base_url,
                        **kwargs,
                    )
                # 纯文本格式
                else:
                    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

            # 定义嵌入函数
            embedding_func = EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=self.api_key,
                    base_url=self.base_url,
                ),
            )

            # 初始化 RAGAnything
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )

            print("✅ RAG-Anything 初始化成功")
            return rag

        except Exception as e:
            print(f"❌ RAG-Anything 初始化失败: {str(e)}")
            return None

    async def test_document_processing(self, rag):
        """测试文档处理"""
        print("\n测试文档处理...")

        # 创建一个测试文档
        test_doc_path = self.create_advanced_test_document()
        if not test_doc_path:
            print("❌ 无法创建测试文档")
            return False

        try:
            # 处理文档
            output_dir = self.output_dir / "processed_document"
            output_dir.mkdir(exist_ok=True)

            start_time = time.time()
            await rag.process_document_complete(
                file_path=str(test_doc_path), output_dir=str(output_dir), parse_method="auto"
            )
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"文档处理耗时: {execution_time:.2f} 秒")

            # 列出输出文件
            output_files = list(output_dir.rglob("*"))
            print(f"输出文件数量: {len(output_files)}")
            for file in output_files:
                if file.is_file():
                    print(f"  - {file.relative_to(output_dir)}")

            print("✅ 文档处理成功")
            return True

        except Exception as e:
            print(f"❌ 文档处理失败: {str(e)}")
            return False

    async def test_content_insertion(self, rag):
        """测试内容列表插入"""
        print("\n测试内容列表插入...")

        try:
            # 示例：来自外部源的预解析内容列表
            content_list = [
                {
                    "type": "text",
                    "text": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                    "page_idx": 0,
                },
                {
                    "type": "text",
                    "text": "机器学习是人工智能的一个子集，它使用算法从数据中学习并做出预测或决策。",
                    "page_idx": 0,
                },
                {
                    "type": "table",
                    "table_body": (
                        "| 方法 | 准确率 | F1分数 |\n"
                        "|------|--------|--------|\n"
                        "| 深度学习 | 95.2% | 0.94 |\n"
                        "| 传统方法 | 87.3% | 0.85 |"
                    ),
                    "table_caption": ["表1：性能对比"],
                    "table_footnote": ["测试数据集结果"],
                    "page_idx": 1,
                },
                {
                    "type": "equation",
                    "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
                    "text": "贝叶斯概率公式",
                    "page_idx": 2,
                },
                {"type": "text", "text": "总之，机器学习方法在各种任务中都表现出优越的性能。", "page_idx": 3},
            ]

            start_time = time.time()
            await rag.insert_content_list(
                content_list=content_list,
                file_path="ai_overview.pdf",
                split_by_character=None,
                split_by_character_only=False,
                doc_id="ai-overview-doc",
                display_stats=True,
            )
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"内容列表插入耗时: {execution_time:.2f} 秒")

            print("✅ 内容列表插入成功")
            return True

        except Exception as e:
            print(f"❌ 内容列表插入失败: {str(e)}")
            return False

    async def test_text_query(self, rag):
        """测试文本查询"""
        print("\n测试文本查询...")

        try:
            # 执行查询
            query = "机器学习与人工智能的关系是什么？"
            print(f"查询: {query}")

            start_time = time.time()
            result = await rag.aquery(query, mode="hybrid")
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"查询耗时: {execution_time:.2f} 秒")

            print("查询结果:")
            print(result)

            print("✅ 文本查询成功")
            return True

        except Exception as e:
            print(f"❌ 文本查询失败: {str(e)}")
            return False

    async def test_multimodal_query(self, rag):
        """测试多模态查询"""
        print("\n测试多模态查询...")

        try:
            # 创建一个示例表格
            table_content = {
                "type": "table",
                "table_data": """模型,准确率,速度,参数量
                        GPT-4,95.2%,120ms,1.8T
                        Claude-3,94.8%,150ms,未知
                        Gemini-Pro,93.5%,100ms,未知""",
                "table_caption": "大语言模型性能对比",
            }

            query = "分析这些大语言模型的性能数据，并解释它们在处理不同任务时的优缺点"
            print(f"查询: {query}")

            start_time = time.time()
            result = await rag.aquery_with_multimodal(query, multimodal_content=[table_content], mode="hybrid")
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"多模态查询耗时: {execution_time:.2f} 秒")

            print("查询结果:")
            print(result)

            print("✅ 多模态查询成功")
            return True

        except Exception as e:
            print(f"❌ 多模态查询失败: {str(e)}")
            return False

    def create_advanced_test_document(self):
        """创建一个高级测试文档"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.colors import black, blue, red
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.pdfgen import canvas
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

            # 创建 PDF 文档路径
            doc_path = self.test_files_dir / "advanced_test_document.pdf"

            # 创建文档
            doc = SimpleDocTemplate(str(doc_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # 添加标题
            title = Paragraph("RAG-Anything 高级测试文档", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 12))

            # 添加段落
            content = """
            <b>人工智能与机器学习概述</b><br/>
            人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
            这些任务包括学习、推理、问题解决、感知和语言理解。<br/><br/>

            <b>机器学习的核心概念</b><br/>
            机器学习是人工智能的一个子集，它使用算法从数据中学习并做出预测或决策。
            与传统编程方法不同，机器学习系统通过训练数据来改进其性能，而无需显式编程。
            """

            para = Paragraph(content, styles["Normal"])
            story.append(para)
            story.append(Spacer(1, 12))

            # 添加表格
            data = [
                ["方法", "准确率", "F1分数", "训练时间"],
                ["深度学习", "95.2%", "0.94", "120小时"],
                ["传统机器学习", "87.3%", "0.85", "24小时"],
                ["统计方法", "82.1%", "0.79", "8小时"],
            ]

            table = Table(data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 14),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(table)
            story.append(Spacer(1, 12))

            # 添加数学公式
            formula_content = """
            <b>贝叶斯概率公式</b><br/>
            P(d|q) = P(q|d) * P(d) / P(q)<br/><br/>

            这个公式描述了在给定查询 q 的情况下，文档 d 相关性的条件概率。
            它广泛应用于信息检索和自然语言处理领域。
            """

            formula_para = Paragraph(formula_content, styles["Normal"])
            story.append(formula_para)
            story.append(Spacer(1, 12))

            # 添加更多内容
            more_content = """
            <b>深度学习的发展</b><br/>
            深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次表示。
            自2012年以来，深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。<br/><br/>

            <b>应用领域</b><br/>
            1. 计算机视觉：图像分类、目标检测、人脸识别<br/>
            2. 自然语言处理：机器翻译、情感分析、问答系统<br/>
            3. 语音识别：语音转文本、语音合成<br/>
            4. 推荐系统：个性化推荐、协同过滤<br/>
            5. 医疗诊断：疾病预测、医学影像分析
            """

            more_para = Paragraph(more_content, styles["Normal"])
            story.append(more_para)
            story.append(Spacer(1, 12))

            # 添加结论
            conclusion = """
            <b>结论</b><br/>
            人工智能和机器学习技术正在快速发展，并在各个领域展现出巨大的潜力。
            随着计算能力的提高和算法的改进，我们可以预期未来会有更多创新的应用。
            """

            conclusion_para = Paragraph(conclusion, styles["Normal"])
            story.append(conclusion_para)

            # 构建文档
            doc.build(story)

            print(f"✅ 高级测试文档已创建: {doc_path}")
            return doc_path

        except ImportError:
            print("❌ reportlab 库未安装，无法创建 PDF 文档")
            print("请运行: uv add reportlab")
            return None
        except Exception as e:
            print(f"❌ 创建高级测试文档失败: {str(e)}")
            return None

    async def run_all_tests(self):
        """运行所有测试"""
        print("开始 RAG-Anything 高级功能测试")
        print("=" * 60)

        test_results = []

        # 测试 1: RAG-Anything 初始化
        rag = await self.test_raganything_initialization()
        test_results.append(("RAG-Anything 初始化", rag is not None))

        if rag is None:
            print("\n⚠️ RAG-Anything 初始化失败，跳过后续测试")
            print("请确保已正确设置 API 密钥和相关依赖")
        else:
            # 测试 2: 内容列表插入
            insertion_result = await self.test_content_insertion(rag)
            test_results.append(("内容列表插入", insertion_result))

            # 测试 3: 文本查询
            text_query_result = await self.test_text_query(rag)
            test_results.append(("文本查询", text_query_result))

            # 测试 4: 多模态查询
            multimodal_query_result = await self.test_multimodal_query(rag)
            test_results.append(("多模态查询", multimodal_query_result))

            # 测试 5: 文档处理
            doc_processing_result = await self.test_document_processing(rag)
            test_results.append(("文档处理", doc_processing_result))

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
        results_file = self.test_dir / "advanced_test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "api_key_set": bool(self.api_key and self.api_key != "sk-test-key-placeholder"),
                    "base_url": self.base_url,
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
    # 可以通过环境变量或命令行参数设置 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("⚠️ 未设置 OPENAI_API_KEY 环境变量")
        print("某些测试可能会失败")
        print("可以通过以下方式设置:")
        print("export OPENAI_API_KEY=your_api_key")
        print("或者在运行脚本时提供:")
        print("OPENAI_API_KEY=your_api_key python test_advanced_functionality.py")

    tester = RAGAnythingAdvancedTester(api_key=api_key, base_url=base_url)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
