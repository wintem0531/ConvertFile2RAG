# ConvertFile2RAG

一个用于将各种文件格式转换为 RAG（检索增强生成）系统的工具包。

## 项目结构

```
ConvertFile2RAG/
├── modes/                          # 各种文件处理模块
│   ├── RAGAnything/                # RAG-Anything 开源库测试模块
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── install_dependencies.py
│   │   ├── test_basic_functionality.py
│   │   ├── test_advanced_functionality.py
│   │   ├── example_usage.py
│   │   └── README.md
│   ├── Unstructured/               # Unstructured 文件处理模块
│   ├── imageTool/                   # 图像处理工具
│   ├── ocrTool/                     # OCR 工具
│   └── pdfConverter/                # PDF 转换工具
├── workflow/                        # 工作流相关文件
└── run_raganything_tests.py       # RAGAnything 测试运行脚本
```

## RAG-Anything 测试模块

`modes/RAGAnything` 模块用于测试 [RAG-Anything](https://github.com/HKUDS/RAG-Anything) 开源库的效果。RAG-Anything 是一个综合性的多模态文档处理 RAG 系统，能够处理包含文本、图像、表格和公式等多模态内容的复杂文档。

### 快速开始

1. 安装依赖：

```bash
python run_raganything_tests.py --action install
```

2. 基本功能测试：

```bash
python run_raganything_tests.py --action test-basic
```

3. 高级功能测试（需要 API 密钥）：

```bash
python run_raganything_tests.py --action test-advanced --api-key YOUR_API_KEY
```

4. 特定文档测试（测试特定文档的页面范围）：

```bash
python run_raganything_tests.py --action test-doc --api-key YOUR_API_KEY
```

5. 查看使用示例：

```bash
python run_raganything_tests.py --action example --api-key YOUR_API_KEY
```

6. 运行所有测试：

```bash
python run_raganything_tests.py --action all --api-key YOUR_API_KEY
```

### 环境变量

对于高级测试和示例，需要设置以下环境变量：

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_base_url  # 可选
```

### 详细文档

更多详细信息请查看 [modes/RAGAnything/README.md](modes/RAGAnything/README.md)。

## 其他模块

- **Unstructured**: 基于 Unstructured.io 的文件处理模块
- **imageTool**: 图像处理工具
- **ocrTool**: OCR（光学字符识别）工具
- **pdfConverter**: PDF 转换工具

## 工作流

`workflow` 目录包含工作流相关的文件，用于处理复杂的文件转换流程。

## 许可证

本项目遵循 MIT 许可证。