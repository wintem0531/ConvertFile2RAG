#!/usr/bin/env python3
"""
RAG-Anything 特定文档测试

该脚本测试 RAG-Anything 处理特定文档 "test_file/input/齊系文字編.pdf" 的第20-25页。
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


class RAGAnythingDocumentTester:
    """RAG-Anything 特定文档测试类"""

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

    def get_document_path(self):
        """获取测试文档路径"""
        # 首先检查项目根目录下的test_file目录
        doc_path = project_root / "test_file" / "input" / "齊系文字編.pdf"

        if doc_path.exists():
            return doc_path

        # 如果不存在，检查其他可能的位置
        possible_paths = [
            Path("/Users/songtao/PycharmProjects/ConvertFile2RAG/test_file/input/齊系文字編.pdf"),
            Path("./test_file/input/齊系文字編.pdf"),
            Path("../test_file/input/齊系文字編.pdf"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    async def test_document_existence(self):
        """测试文档是否存在"""
        print("\n测试文档是否存在...")

        doc_path = self.get_document_path()
        if doc_path and doc_path.exists():
            print(f"✅ 文档存在: {doc_path}")
            return doc_path
        else:
            print("❌ 文档不存在: 齊系文字編.pdf")
            print("请确保文档位于以下路径之一:")
            print(f"  1. {project_root}/test_file/input/齊系文字編.pdf")
            print("  2. /Users/songtao/PycharmProjects/ConvertFile2RAG/test_file/input/齊系文字編.pdf")
            print("  3. ./test_file/input/齊系文字編.pdf")
            return None

    async def test_mineru_page_range_parsing(self, doc_path):
        """测试 MinerU 页面范围解析"""
        print("\n测试 MinerU 页面范围解析...")

        try:
            import subprocess

            output_dir = self.output_dir / "mineru_page_range"
            output_dir.mkdir(exist_ok=True)

            # 使用 MinerU 解析文档的特定页面 (20-25)
            # MinerU 页码从0开始，所以20-25页对应19-24
            command = f"mineru -p {doc_path} -o {output_dir} -m auto -b pipeline --start-page 19 --end-page 24"
            print(f"执行命令: {command}")

            start_time = time.time()
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"解析耗时: {execution_time:.2f} 秒")

            if result.returncode == 0:
                print("✅ MinerU 页面范围解析成功")
                print(f"输出目录: {output_dir}")

                # 列出输出文件
                output_files = list(output_dir.rglob("*"))
                print(f"输出文件数量: {len(output_files)}")
                for file in output_files:
                    if file.is_file():
                        print(f"  - {file.relative_to(output_dir)}")

                return True, output_dir
            else:
                print(f"❌ MinerU 页面范围解析失败: {result.stderr}")
                return False, None

        except Exception as e:
            print(f"❌ 页面范围解析测试失败: {str(e)}")
            return False, None

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

    async def test_document_page_range_processing(self, rag, doc_path):
        """测试文档页面范围处理"""
        print("\n测试文档页面范围处理...")

        try:
            # 处理文档的特定页面
            output_dir = self.output_dir / "processed_page_range"
            output_dir.mkdir(exist_ok=True)

            start_time = time.time()
            await rag.process_document_complete(
                file_path=str(doc_path),
                output_dir=str(output_dir),
                parse_method="auto",
                start_page=19,  # 第20页 (从0开始)
                end_page=24,  # 第25页 (从0开始)
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

            print("✅ 文档页面范围处理成功")
            return True

        except Exception as e:
            print(f"❌ 文档页面范围处理失败: {str(e)}")
            return False

    async def test_content_extraction(self, output_dir):
        """测试内容提取"""
        print("\n测试内容提取...")

        try:
            # 查找解析后的内容文件
            content_files = []

            # 查找content_list.json文件
            for file in output_dir.rglob("*.json"):
                if "content_list" in file.name:
                    content_files.append(file)

            if not content_files:
                print("❌ 未找到内容列表文件")
                return False

            # 分析每个内容文件
            for content_file in content_files:
                print(f"\n分析内容文件: {content_file.relative_to(output_dir)}")

                try:
                    with open(content_file, encoding="utf-8") as f:
                        content_data = json.load(f)

                    if isinstance(content_data, list):
                        print(f"内容项数量: {len(content_data)}")

                        # 分析内容类型
                        content_types = {}
                        page_indices = set()

                        for item in content_data:
                            item_type = item.get("type", "unknown")
                            content_types[item_type] = content_types.get(item_type, 0) + 1

                            page_idx = item.get("page_idx", None)
                            if page_idx is not None:
                                page_indices.add(page_idx)

                        print("内容类型分布:")
                        for c_type, count in content_types.items():
                            print(f"  {c_type}: {count}")

                        if page_indices:
                            print(f"页面范围: {min(page_indices)}-{max(page_indices)}")

                        # 显示部分内容示例
                        print("\n内容示例:")
                        for i, item in enumerate(content_data[:3]):  # 只显示前3个
                            item_type = item.get("type", "unknown")
                            if item_type == "text":
                                text = (
                                    item.get("text", "")[:100] + "..."
                                    if len(item.get("text", "")) > 100
                                    else item.get("text", "")
                                )
                                print(f"  [{i}] 文本: {text}")
                            elif item_type == "image":
                                caption = item.get("image_caption", "")
                                if isinstance(caption, list) and caption:
                                    caption = caption[0]
                                print(
                                    f"  [{i}] 图像: {caption[:50]}..."
                                    if len(str(caption)) > 50
                                    else f"  [{i}] 图像: {caption}"
                                )
                            elif item_type == "table":
                                caption = item.get("table_caption", "")
                                if isinstance(caption, list) and caption:
                                    caption = caption[0]
                                print(
                                    f"  [{i}] 表格: {caption[:50]}..."
                                    if len(str(caption)) > 50
                                    else f"  [{i}] 表格: {caption}"
                                )
                            elif item_type == "equation":
                                latex = item.get("latex", "")
                                print(
                                    f"  [{i}] 公式: {latex[:50]}..."
                                    if len(str(latex)) > 50
                                    else f"  [{i}] 公式: {latex}"
                                )
                            else:
                                print(f"  [{i}] {item_type}")

                except Exception as e:
                    print(f"❌ 解析内容文件失败: {str(e)}")

            print("✅ 内容提取成功")
            return True

        except Exception as e:
            print(f"❌ 内容提取失败: {str(e)}")
            return False

    async def run_all_tests(self):
        """运行所有测试"""
        print("开始 RAG-Anything 特定文档测试")
        print("=" * 60)

        test_results = []

        # 测试 1: 文档存在性
        doc_path = await self.test_document_existence()
        test_results.append(("文档存在性", doc_path is not None))

        if not doc_path:
            print("\n❌ 测试文档不存在，跳过后续测试")
            return

        # 测试 2: MinerU 页面范围解析
        parse_success, output_dir = await self.test_mineru_page_range_parsing(doc_path)
        test_results.append(("MinerU 页面范围解析", parse_success))

        # 测试 3: RAG-Anything 初始化
        rag = await self.test_raganything_initialization()
        test_results.append(("RAG-Anything 初始化", rag is not None))

        if rag is not None:
            # 测试 4: 文档页面范围处理
            doc_processing_result = await self.test_document_page_range_processing(rag, doc_path)
            test_results.append(("文档页面范围处理", doc_processing_result))

        if output_dir:
            # 测试 5: 内容提取
            content_extraction_result = await self.test_content_extraction(output_dir)
            test_results.append(("内容提取", content_extraction_result))

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
        results_file = self.test_dir / "document_test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "document_path": str(doc_path) if doc_path else None,
                    "page_range": "20-25",
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
        print("OPENAI_API_KEY=your_api_key python test_specific_document.py")

    tester = RAGAnythingDocumentTester(api_key=api_key, base_url=base_url)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
