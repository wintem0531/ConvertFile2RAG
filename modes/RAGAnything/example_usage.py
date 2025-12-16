#!/usr/bin/env python3
"""
RAG-Anything 使用示例

该脚本展示了如何使用 RAG-Anything 处理文档和进行查询的基本示例。
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


async def example_basic_usage():
    """基本使用示例"""
    print("RAG-Anything 基本使用示例")
    print("=" * 50)

    try:
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig

        # 设置 API 密钥
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            print("⚠️ 未设置 OPENAI_API_KEY，示例仅展示代码结构")
            print("请设置环境变量: export OPENAI_API_KEY=your_api_key")
            return

        # 1. 创建配置
        print("\n1. 创建配置...")
        config = RAGAnythingConfig(
            working_dir="./output/rag_storage",
            parser="mineru",  # 可选: "mineru" 或 "docling"
            parse_method="auto",  # 可选: "auto", "ocr", "txt"
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        print("✅ 配置创建成功")

        # 2. 定义模型函数
        print("\n2. 定义模型函数...")

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

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
                    api_key=api_key,
                    base_url=base_url,
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
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # 纯文本格式
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
        print("✅ 模型函数定义成功")

        # 3. 初始化 RAGAnything
        print("\n3. 初始化 RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        print("✅ RAGAnything 初始化成功")

        # 4. 处理文档
        print("\n4. 处理文档...")
        # 替换为实际文档路径
        doc_path = "path/to/your/document.pdf"
        output_dir = "./output/processed_document"

        if not os.path.exists(doc_path):
            print(f"⚠️ 文档不存在: {doc_path}")
            print("请替换为实际的文档路径")
        else:
            await rag.process_document_complete(file_path=doc_path, output_dir=output_dir, parse_method="auto")
            print(f"✅ 文档处理完成: {output_dir}")

        # 5. 查询处理后的内容
        print("\n5. 查询处理后的内容...")

        # 纯文本查询
        query = "这个文档的主要内容是什么？"
        print(f"查询: {query}")

        # text_result = await rag.aquery(query, mode="hybrid")
        # print("文本查询结果:", text_result)
        print("(注释掉实际查询，因为需要先处理文档)")

        print("\n示例代码展示完成")

    except ImportError as e:
        print(f"❌ 导入错误: {str(e)}")
        print("请先运行 install_dependencies.py 安装依赖")
    except Exception as e:
        print(f"❌ 示例执行失败: {str(e)}")


async def example_content_insertion():
    """内容列表插入示例"""
    print("\n\nRAG-Anything 内容列表插入示例")
    print("=" * 50)

    try:
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig

        # 设置 API 密钥
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            print("⚠️ 未设置 OPENAI_API_KEY，示例仅展示代码结构")
            print("请设置环境变量: export OPENAI_API_KEY=your_api_key")
            return

        # 1. 创建配置
        print("\n1. 创建配置...")
        config = RAGAnythingConfig(
            working_dir="./output/rag_storage_content",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 2. 定义模型函数
        print("\n2. 定义模型函数...")

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
        ):
            # 视觉模型函数实现
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
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
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )

        # 3. 初始化 RAGAnything
        print("\n3. 初始化 RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # 4. 创建内容列表
        print("\n4. 创建内容列表...")
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
        print("✅ 内容列表创建成功")

        # 5. 插入内容列表
        print("\n5. 插入内容列表...")
        await rag.insert_content_list(
            content_list=content_list,
            file_path="ai_overview.pdf",
            split_by_character=None,
            split_by_character_only=False,
            doc_id="ai-overview-doc",
            display_stats=True,
        )
        print("✅ 内容列表插入成功")

        # 6. 查询插入的内容
        print("\n6. 查询插入的内容...")
        query = "机器学习与人工智能的关系是什么？"
        print(f"查询: {query}")

        # result = await rag.aquery(query, mode="hybrid")
        # print("查询结果:", result)
        print("(注释掉实际查询，以避免API调用)")

        print("\n内容列表插入示例完成")

    except ImportError as e:
        print(f"❌ 导入错误: {str(e)}")
        print("请先运行 install_dependencies.py 安装依赖")
    except Exception as e:
        print(f"❌ 示例执行失败: {str(e)}")


async def example_multimodal_query():
    """多模态查询示例"""
    print("\n\nRAG-Anything 多模态查询示例")
    print("=" * 50)

    try:
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig

        # 设置 API 密钥
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            print("⚠️ 未设置 OPENAI_API_KEY，示例仅展示代码结构")
            print("请设置环境变量: export OPENAI_API_KEY=your_api_key")
            return

        # 1. 创建配置
        print("\n1. 创建配置...")
        config = RAGAnythingConfig(
            working_dir="./output/rag_storage_multimodal",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 2. 定义模型函数
        print("\n2. 定义模型函数...")

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
        ):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
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
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )

        # 3. 初始化 RAGAnything
        print("\n3. 初始化 RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # 4. 创建基本内容（在实际应用中，这些内容可能来自文档解析）
        print("\n4. 插入基本内容...")
        content_list = [
            {
                "type": "text",
                "text": "大语言模型是当前人工智能研究的热点，它们在各种自然语言处理任务中表现出色。",
                "page_idx": 0,
            },
            {
                "type": "text",
                "text": "不同的模型有不同的特点和适用场景，选择合适的模型对于应用的成功至关重要。",
                "page_idx": 0,
            },
        ]

        await rag.insert_content_list(
            content_list=content_list, file_path="llm_overview.pdf", doc_id="llm-overview-doc"
        )
        print("✅ 基本内容插入成功")

        # 5. 多模态查询
        print("\n5. 多模态查询...")

        # 包含表格数据的查询
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
        print("表格数据:")
        print(table_content["table_data"])

        # result = await rag.aquery_with_multimodal(
        #     query,
        #     multimodal_content=[table_content],
        #     mode="hybrid"
        # )
        # print("查询结果:", result)
        print("(注释掉实际查询，以避免API调用)")

        print("\n多模态查询示例完成")

    except ImportError as e:
        print(f"❌ 导入错误: {str(e)}")
        print("请先运行 install_dependencies.py 安装依赖")
    except Exception as e:
        print(f"❌ 示例执行失败: {str(e)}")


async def main():
    """主函数"""
    print("RAG-Anything 使用示例")
    print("=" * 50)

    # 检查 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ 未设置 OPENAI_API_KEY 环境变量")
        print("示例将仅展示代码结构，不会执行实际的 API 调用")
        print("设置方法: export OPENAI_API_KEY=your_api_key")
        print()

    # 运行示例
    await example_basic_usage()
    await example_content_insertion()
    await example_multimodal_query()

    print("\n\n示例脚本执行完成")
    print("=" * 50)
    print("如需运行实际功能测试，请:")
    print("1. 设置 OPENAI_API_KEY 环境变量")
    print("2. 运行 test_basic_functionality.py 进行基本测试")
    print("3. 运行 test_advanced_functionality.py 进行高级测试")


if __name__ == "__main__":
    asyncio.run(main())
