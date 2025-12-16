# RAG-Anything 测试模块

该模块用于测试 [RAG-Anything](https://github.com/HKUDS/RAG-Anything) 开源库的效果。RAG-Anything 是一个综合性的多模态文档处理 RAG 系统，能够处理包含文本、图像、表格和公式等多模态内容的复杂文档。

## 功能特点

- 📄 多格式文档支持：PDF、Office文档、图像、文本文件
- 🧠 多模态内容分析：针对图像、表格、公式和通用文本内容部署专门的处理器
- 🔗 基于知识图谱索引：实现自动化实体提取和关系构建
- ⚡ 灵活的处理架构：支持基于 MinerU 的智能解析模式和直接多模态内容插入模式
- 🎯 跨模态检索机制：实现跨文本和多模态内容的智能检索

## 文件结构

```
RAGAnything/
├── __init__.py                     # 模块初始化文件
├── __main__.py                     # 模块入口文件
├── install_dependencies.py         # 依赖安装脚本
├── test_basic_functionality.py     # 基本功能测试
├── test_advanced_functionality.py  # 高级功能测试
├── test_specific_document.py      # 特定文档测试
├── example_usage.py                # 使用示例
├── README.md                       # 说明文档
├── test_files/                     # 测试文件目录
│   └── (自动生成)
└── output/                         # 输出目录
    ├── (自动生成)
    └── rag_storage/                # RAG 存储目录
        └── (自动生成)
```

## 安装与设置

### 1. 安装依赖

首先运行依赖安装脚本：

```bash
python install_dependencies.py
```

或者手动安装：

```bash
# 安装基础 RAG-Anything
uv add raganything

# 安装扩展依赖
uv add 'raganything[all]'

# 或者使用 pip
pip install 'raganything[all]'
```

### 2. 安装 MinerU

RAG-Anything 依赖 MinerU 进行文档解析。请参考 [MinerU 安装指南](https://github.com/opendatalab/MinerU/blob/master/README_zh-CN.md)。

检查 MinerU 安装：

```bash
mineru --version
```

### 3. 安装 LibreOffice（可选）

如果需要处理 Office 文档，请安装 LibreOffice：

- **macOS**: `brew install --cask libreoffice`
- **Ubuntu/Debian**: `sudo apt-get install libreoffice`
- **Windows**: 从 [官网](https://www.libreoffice.org/download/download/) 下载安装

### 4. 设置环境变量

为了进行完整的功能测试，需要设置 OpenAI API 密钥：

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_base_url  # 可选
```

## 使用方法

### 基本功能测试

运行基本功能测试，检查依赖安装和基本配置：

```bash
python test_basic_functionality.py
```

基本功能测试包括：
- RAG-Anything 导入测试
- MinerU 可用性测试
- 示例文档处理配置测试
- MinerU 直接解析测试

### 高级功能测试

运行高级功能测试，测试完整的 RAG 流程：

```bash
python test_advanced_functionality.py
```

高级功能测试包括：
- RAG-Anything 初始化
- 内容列表插入
- 文本查询
- 多模态查询
- 文档处理

### 特定文档测试

运行特定文档测试，测试特定文档和页面范围的处理：

```bash
python test_specific_document.py
```

特定文档测试包括：
- 文档存在性检查
- MinerU 页面范围解析 (测试 "test_file/input/齊系文字編.pdf" 的第20-25页)
- RAG-Anything 初始化
- 文档页面范围处理
- 内容提取和分析

通过模块入口运行：
```bash
python -m modes.RAGAnything --action test-doc --api-key YOUR_API_KEY
```

### 使用示例

查看 RAG-Anything 的使用示例：

```bash
python example_usage.py
```

使用示例包括：
- 基本使用示例
- 内容列表插入示例
- 多模态查询示例

## 测试结果

测试完成后，结果将保存在以下文件中：

- `test_results.json`: 基本功能测试结果
- `advanced_test_results.json`: 高级功能测试结果
- `output/`: 各种处理输出和测试文件

## 示例代码

### 基本使用

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # 设置 API 配置
    api_key = "your-api-key"
    base_url = "your-base-url"  # 可选

    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # 选择解析器：mineru 或 docling
        parse_method="auto",  # 解析方法：auto, ocr 或 txt
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
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # 定义视觉模型函数用于图像处理
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # 实现视觉模型函数
        pass

    # 定义嵌入函数
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

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # 处理文档
    await rag.process_document_complete(
        file_path="path/to/your/document.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # 查询处理后的内容
    result = await rag.aquery(
        "文档的主要内容是什么？",
        mode="hybrid"
    )
    print("查询结果:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 内容列表插入

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig

async def insert_content_example():
    # 初始化 RAGAnything (同上)
    
    # 示例：来自外部源的预解析内容列表
    content_list = [
        {
            "type": "text",
            "text": "人工智能（AI）是计算机科学的一个分支...",
            "page_idx": 0
        },
        {
            "type": "image",
            "img_path": "/path/to/image.jpg",  # 注意：必须使用绝对路径
            "image_caption": ["图1：AI 发展历程"],
            "image_footnote": ["来源：研究机构"],
            "page_idx": 1
        },
        {
            "type": "table",
            "table_body": "| 方法 | 准确率 | F1分数 |\\n|------|--------|--------|\\n| 深度学习 | 95.2% | 0.94 |",
            "table_caption": ["表1：性能对比"],
            "table_footnote": ["测试数据集结果"],
            "page_idx": 2
        },
        {
            "type": "equation",
            "latex": "P(d|q) = \\\\frac{P(q|d) \\\\cdot P(d)}{P(q)}",
            "text": "贝叶斯概率公式",
            "page_idx": 3
        }
    ]

    # 直接插入内容列表
    await rag.insert_content_list(
        content_list=content_list,
        file_path="research_paper.pdf",  # 用于引用的参考文件名
        split_by_character=None,         # 可选的文本分割
        split_by_character_only=False,   # 可选的文本分割模式
        doc_id=None,                     # 可选的自定义文档ID
        display_stats=True               # 显示内容统计信息
    )

if __name__ == "__main__":
    asyncio.run(insert_content_example())
```

### 多模态查询

```python
# 纯文本查询
text_result = await rag.aquery("你的问题", mode="hybrid")

# VLM增强查询（当文档包含图片时，VLM可以直接查看和分析图片）
vlm_result = await rag.aquery(
    "分析文档中的图表和数据",
    mode="hybrid"
    # vlm_enhanced=True 当vision_model_func可用时自动设置
)

# 多模态查询 - 包含特定多模态内容分析的增强查询
table_result = await rag.aquery_with_multimodal(
    "比较这些性能指标与文档内容",
    multimodal_content=[{
        "type": "table",
        "table_data": """方法,准确率,速度
                        LightRAG,95.2%,120ms
                        传统方法,87.3%,180ms""",
        "table_caption": "性能对比"
    }],
    mode="hybrid"
)
```

## 常见问题

### Q: MinerU 安装失败怎么办？

A: 请检查 Python 版本（建议 3.9+）和网络连接。可以尝试使用国内镜像源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple magic-pdf[full]
```

### Q: 处理 PDF 文档时出现乱码怎么办？

A: 尝试指定解析语言参数：

```python
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output",
    parse_method="auto",
    lang="ch"  # 指定中文文档
)
```

### Q: 如何提高文档处理速度？

A: 可以使用 GPU 加速：

```python
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output",
    parse_method="auto",
    device="cuda"  # 使用 GPU 加速
)
```

## 参考资料

- [RAG-Anything 官方文档](https://github.com/HKUDS/RAG-Anything)
- [MinerU 官方文档](https://github.com/opendatalab/MinerU)
- [LightRAG 官方文档](https://github.com/HKUDS/LightRAG)

## 许可证

本测试模块遵循原项目的 MIT 许可证。