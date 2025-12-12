# MinerU 工具函数封装

本模块提供了对 MinerU 库的封装，简化了 PDF 文档解析和内容提取的操作。

## 功能特性

- 📄 **PDF/图像解析**：支持 PDF 和图像文件（PNG、JPEG 等）的解析
- 📝 **内容提取**：提取文本、标题、图片、表格等结构化内容
- 📊 **多种输出格式**：支持内容列表、中间 JSON、纯文本等多种输出格式
- 🎯 **页面范围选择**：支持指定解析页面范围
- 🌐 **多语言支持**：支持中文、英文等多种语言

## 主要函数

### 1. `extract_content_list_from_pdf`

从 PDF 或图像文件提取内容列表，返回结构化数据。

```python
from modes.mineru.mineru_util import extract_content_list_from_pdf

content_list = extract_content_list_from_pdf(
    pdf_path="path/to/your/file.pdf",  # 支持图像文件
    page_range=(1, 3),  # 解析第1-3页，None表示全部页面
    output_dir="output",  # 输出目录，None表示使用临时目录
    lang="ch",  # 语言：ch=中文，en=英文
    backend="vlm-mlx-engine",  # 后端引擎
    save_result=True  # 是否保存结果到文件
)
```

### 2. `extract_text_from_pdf`

直接从 PDF 或图像文件提取纯文本内容。

```python
from modes.mineru.mineru_util import extract_text_from_pdf

text_content = extract_text_from_pdf(
    pdf_path="path/to/your/file.pdf",
    page_range=(1, 2)  # 只提取第1-2页的文本
)

print(text_content)
```

### 3. `parse_pdf_to_middle_json`

获取 MinerU 的中间 JSON 结果，包含更详细的解析信息。

```python
from modes.mineru.mineru_util import parse_pdf_to_middle_json

middle_json = parse_pdf_to_middle_json(
    pdf_path="path/to/your/file.pdf",
    page_range=(1, 1)  # 只解析第一页
)

# 访问详细的解析信息
pdf_info = middle_json.get("pdf_info", [])
for page in pdf_info:
    preproc_blocks = page.get("preproc_blocks", [])
    print(f"页面包含 {len(preproc_blocks)} 个预处理块")
```

### 4. `extract_text_blocks_from_content_list`

从内容列表中提取文本块，保留位置和类型信息。

```python
from modes.mineru.mineru_util import extract_text_blocks_from_content_list

# 假设已有 content_list
text_blocks = extract_text_blocks_from_content_list(content_list)

for block in text_blocks:
    text = block.get("text", "")
    block_type = block.get("type", "")
    bbox = block.get("bbox", [])
    page_idx = block.get("page_idx", 0)
    
    print(f"类型: {block_type}, 页面: {page_idx+1}")
    print(f"文本: {text[:50]}...")
    print(f"位置: {bbox}")
```

### 5. `extract_images_from_content_list`

从内容列表中提取图片信息。

```python
from modes.mineru.mineru_util import extract_images_from_content_list

# 假设已有 content_list
images = extract_images_from_content_list(
    content_list, 
    output_dir="output"  # 提供输出目录以获取绝对路径
)

for img in images:
    img_path = img.get("img_path", "")
    captions = img.get("image_caption", [])
    absolute_path = img.get("absolute_path", "")
    
    print(f"图片路径: {img_path}")
    print(f"图片说明: {captions}")
    if absolute_path:
        print(f"绝对路径: {absolute_path}")
```

## 使用示例

### 基本用法

```python
from pathlib import Path
from modes.mineru.mineru_util import extract_content_list_from_pdf, extract_text_from_pdf

# 文件路径（支持 PDF 和图像）
file_path = Path("path/to/your/document.pdf")

# 提取结构化内容
content_list = extract_content_list_from_pdf(
    pdf_path=file_path,
    page_range=(1, 3),  # 处理第1-3页
    output_dir="output",
    save_result=True
)

# 提取纯文本
text = extract_text_from_pdf(
    pdf_path=file_path,
    page_range=(1, 3)  # 同样的页面范围
)

print(f"提取了 {len(content_list)} 个内容元素")
print(f"提取了 {len(text)} 个字符的文本")
```

### 高级用法

```python
from modes.mineru.mineru_util import (
    parse_pdf_to_middle_json,
    extract_text_blocks_from_content_list,
    extract_images_from_content_list
)

# 获取详细解析结果
middle_json = parse_pdf_to_middle_json(
    pdf_path="document.pdf",
    backend="vlm-mlx-engine",  # 使用不同的后端
    formula_enable=True,  # 启用公式识别
    table_enable=True  # 启用表格识别
)

# 提取特定类型的内容
content_list = middle_json.get("content_list", [])

# 提取所有文本块
text_blocks = extract_text_blocks_from_content_list(content_list)

# 提取所有图片
images = extract_images_from_content_list(
    content_list, 
    output_dir="output"
)

# 筛选标题
headers = [block for block in text_blocks if block.get("type") == "header"]
for header in headers:
    print(f"标题: {header.get('text', '')}")
```

## 页面范围

页面范围使用 1-based 索引：

```python
# 解析所有页面
page_range = None

# 只解析第1页
page_range = (1, 1)

# 解析第2页到第5页
page_range = (2, 5)

# 解析第3页到最后一页
page_range = (3, None)
```

## 后端引擎

支持多种后端引擎：

```python
# 默认后端
backend = "vlm-mlx-engine"

# 其他可用后端
backend = "vlm-transformers"
backend = "vlm-llm-engine"
```

## 输出格式

### 内容列表格式

```json
[
  {
    "type": "text",  // 内容类型: text, image, header, page_number 等
    "text": "文本内容",
    "bbox": [x1, y1, x2, y2],  // 边界框坐标
    "page_idx": 0  // 页面索引（0-based）
  },
  {
    "type": "image",
    "img_path": "images/example.jpg",
    "image_caption": ["图片说明"],
    "bbox": [x1, y1, x2, y2],
    "page_idx": 0
  }
]
```

### 中间 JSON 格式

中间 JSON 包含更详细的解析信息，包括预处理块、段落块等，适用于高级分析和自定义处理。

## 注意事项

1. **文件路径**：支持 PDF 文件和常见图像格式（PNG、JPEG 等）
2. **内存使用**：处理大型文档可能需要较多内存
3. **处理时间**：使用 VLM 后端可能需要较长时间
4. **GPU 加速**：在支持的设备上会自动使用 GPU 加速

## 故障排除

### 常见错误

1. **ModuleNotFoundError**：确保已正确安装 MinerU 库
   ```bash
   uv add mineru
   ```

2. **文件不存在**：检查文件路径是否正确

3. **内存不足**：尝试减少页面范围或使用更小的文件

4. **后端错误**：尝试切换不同的后端引擎

### 性能优化

1. 使用页面范围限制处理区域
2. 对于纯文本提取，禁用图片和表格识别
3. 在支持 MPS/NVIDIA GPU 的环境中，会自动使用加速

## 测试

运行测试脚本验证功能：

```bash
uv run test_mineru_util.py
```

更多示例和测试代码请参考 `test_mineru_util.py` 文件。