# MinerU 标准化输出改进总结

## 改进内容

### 1. 统一的数据结构

创建了标准化的数据类来表示 MinerU 的输出：

- **`BoundingBox`**: 边界框信息，提供宽度、高度和面积属性
- **`ContentBlock`**: 基础内容块抽象类
- **`TextBlock`**: 文本块，包含文本级别信息
- **`ImageBlock`**: 图片块，包含路径、说明和注释
- **`TableBlock`**: 表格块，包含表格内容和 Markdown 格式
- **`EquationBlock`**: 公式块，包含 LaTeX 表达式
- **`MinerUOutput`**: 统一的输出容器，包含所有内容块和元数据

### 2. 统一的 API 接口

新增的主要函数：

- **`parse_pdf()`**: 完整参数的 PDF 解析函数，支持不同后端和返回格式
- **`parse_pdf_simple()`**: 简化版 PDF 解析函数，使用默认参数

### 3. 便捷的访问方法

`MinerUOutput` 类提供了便捷的属性和方法：

- `output.text_blocks`: 获取所有文本块
- `output.image_blocks`: 获取所有图片块
- `output.table_blocks`: 获取所有表格块
- `output.equation_blocks`: 获取所有公式块
- `output.plain_text`: 获取纯文本内容
- `output.markdown`: 获取 Markdown 格式内容
- `output.get_blocks_by_page(page_idx)`: 获取指定页面的内容

### 4. 后端兼容性

支持两种后端，并确保输出格式统一：
- `vlm-mlx-engine`
- `pipeline`

### 5. 向后兼容性

保持了原有 API 的兼容性：
- `extract_text_from_pdf()`: 继续返回纯文本字符串
- `extract_content_list_from_pdf()`: 可选择返回原始格式或新的标准化格式

## 使用示例

### 基本使用

```python
from modes.mineru.mineru_util import parse_pdf_simple

# 解析 PDF
output = parse_pdf_simple("document.pdf")

# 访问文本
for text_block in output.text_blocks:
    print(f"文本: {text_block.text}")
    print(f"级别: {text_block.text_level}")

# 获取纯文本
text = output.plain_text

# 获取 Markdown 格式
markdown = output.markdown
```

### 高级使用

```python
from modes.mineru.mineru_util import parse_pdf

# 详细配置
output = parse_pdf(
    pdf_path="document.pdf",
    page_range=(1, 5),  # 解析第1-5页
    backend="vlm-mlx-engine",
    lang="en",
    return_format="content_list"
)

# 按页面访问内容
page_1_blocks = output.get_blocks_by_page(0)

# 筛选特定内容
headers = [b for b in output.text_blocks if b.text_level > 0]
images = output.image_blocks
```

### 向后兼容使用

```python
# 原有方式仍然有效
from modes.mineru.mineru_util import extract_text_from_pdf

text = extract_text_from_pdf("document.pdf")
```

## 优势

1. **统一接口**: 不论使用哪种后端，输出格式都是一致的
2. **类型安全**: 使用数据类提供类型提示和属性访问
3. **易于使用**: 通过直观的属性名访问不同类型的内容
4. **灵活导出**: 支持纯文本、Markdown 等多种格式
5. **向后兼容**: 现有代码无需修改即可继续使用

## 测试文件

- `test_mineru_standardized_output.py`: 验证标准化输出的测试
- `examples/mineru_standardized_usage.py`: 详细的使用示例

## 注意事项

1. 所有页面索引从 0 开始
2. 边界框坐标使用浮点数，格式为 (x0, y0, x1, y1)
3. 文本级别：0=正文，1=一级标题，2=二级标题，以此类推