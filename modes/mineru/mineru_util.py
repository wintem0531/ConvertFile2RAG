"""MinerU 工具函数封装模块

该模块封装了 MinerU 库的调用，提供了便捷的接口用于 PDF 文档解析和内容提取。
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# PDF to image conversion
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("无法导入 PyMuPDF (fitz)，请安装: pip install pymupdf")

# OpenCV for visualization
try:
    import cv2
    import numpy as np
except ImportError:
    print("警告：无法导入 OpenCV，可视化功能不可用")

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mineru.cli.common import do_parse, read_fn
    from mineru.utils.enum_class import MakeMode
except ImportError:
    raise ImportError("无法导入 MinerU 库，请确保已正确安装 MinerU")


def convert_pdf_to_png(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    page_range: tuple[int, int] | None = None
) -> list[tuple[Path, int, int]]:
    """将PDF转换为PNG图像

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        dpi: 分辨率，默认200
        page_range: 页面范围 (start_page, end_page)，0-based 索引，None 表示全部页面
                   例如: (0, 2) 表示转换第0-2页（前3页）

    Returns:
        List[Tuple[Path, int, int]]: 每页的PNG路径、宽度、高度
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    page_info_list = []

    # 确定页面范围
    total_pages = len(pdf_document)

    if page_range is None:
        # 转换所有页面
        start_page = 0
        end_page = total_pages - 1
    else:
        start_page, end_page = page_range

        # 验证页面范围
        if start_page < 0:
            start_page = 0
        if end_page >= total_pages:
            end_page = total_pages - 1
        if start_page > end_page:
            raise ValueError(f"起始页 {start_page} 不能大于结束页 {end_page}")

    # 转换指定范围的页面
    for page_num in range(start_page, end_page + 1):
        # 获取页面
        page = pdf_document[page_num]

        # 创建变换矩阵以设置DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # 渲染页面为图像
        pix = page.get_pixmap(matrix=mat)

        # 生成文件名
        png_path = output_dir / f"page_{page_num:04d}.png"

        # 保存PNG
        pix.save(png_path)

        # 记录页面信息
        page_info_list.append((png_path, pix.width, pix.height))

    # 关闭PDF文档
    pdf_document.close()

    return page_info_list


class BoundingBox:
    """边界框信息

    支持两种坐标格式：
    1. 绝对坐标：直接使用像素值
    2. 相对坐标：相对于页面尺寸的百分比 (0-1)，会被转换为绝对坐标
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __init__(
        self, x0: float, y0: float, x1: float, y1: float, page_width: int | None = None, page_height: int | None = None
    ):
        """初始化边界框

        Args:
            x0, y0, x1, y1: 坐标值
            page_width: 页面宽度（像素），如果提供且坐标在0-1之间，则进行坐标转换
            page_height: 页面高度（像素），如果提供且坐标在0-1之间，则进行坐标转换
        """
        # 检查是否为相对坐标（0-1之间）
        if page_width and page_height and 0 <= x0 <= 1 and 0 <= x1 <= 1 and 0 <= y0 <= 1 and 0 <= y1 <= 1:
            # 转换为绝对坐标
            self.x0 = x0 * page_width
            self.y0 = y0 * page_height
            self.x1 = x1 * page_width
            self.y1 = y1 * page_height
        else:
            # 直接使用绝对坐标
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        """转换为字典格式"""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class ContentBlock:
    """基础内容块"""

    content: str
    block_type: str  # text, image, table, equation, etc.
    bbox: BoundingBox | None = None
    page_idx: int = 0

    def __str__(self) -> str:
        """自动获取并输出所有属性"""
        attrs = []
        # 遍历对象的所有属性
        for key, value in self.__dict__.items():
            # 跳过私有属性和特殊属性
            if not key.startswith('_'):
                # 对于字符串类型的属性，限制长度以避免过长输出
                if isinstance(value, str) and len(value) > 100:
                    attrs.append(f"{key}={value[:100]}...")
                else:
                    attrs.append(f"{key}={value}")

        # 按照属性名排序，保持输出的一致性
        attrs.sort()

        # 生成类名和属性字符串
        class_name = self.__class__.__name__
        return f"{class_name}({', '.join(attrs)})"


@dataclass
class TextBlock(ContentBlock):
    """文本块"""

    block_type: str = "text"
    text_level: int = 0  # 文本级别（0=body, 1=header, etc.)

    @property
    def text(self) -> str:
        return self.content


@dataclass
class ImageBlock(ContentBlock):
    """图片块"""

    block_type: str = "image"
    img_path: str | None = None
    caption: list[str] = field(default_factory=list)
    footnote: list[str] = field(default_factory=list)
    img_data: bytes | None = None  # 图片的二进制数据
    absolute_path: str | None = None  # 图片的绝对路径

    @property
    def image_caption(self) -> list[str]:
        return self.caption

    @property
    def image_footnote(self) -> list[str]:
        return self.footnote

    def load_image_data(self, base_dir: Path | None = None):
        """加载图片数据到内存

        Args:
            base_dir: 基础目录，如果 img_path 是相对路径，则从此目录加载
        """
        if self.img_path and self.img_data is None:
            img_file_path = Path(self.img_path)

            # 如果是相对路径且有基础目录，则组合成绝对路径
            if base_dir and not img_file_path.is_absolute():
                img_file_path = base_dir / img_file_path

            try:
                if img_file_path.exists():
                    self.img_data = img_file_path.read_bytes()
                    self.absolute_path = str(img_file_path.absolute())
            except Exception as e:
                print(f"警告：无法加载图片 {self.img_path}: {e}")

    def save_image(self, output_path: str | Path):
        """保存图片到指定位置

        Args:
            output_path: 输出文件路径
        """
        if self.img_data:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(self.img_data)
            return str(output_file)
        elif self.absolute_path and Path(self.absolute_path).exists():
            import shutil

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.absolute_path, output_file)
            return str(output_file)
        else:
            print("警告：无法保存图片，没有图片数据")
            return None


@dataclass
class TableBlock(ContentBlock):
    """表格块"""

    block_type: str = "table"
    table_content: list[list[str]] | None = None
    markdown: str | None = None

    @property
    def table_text(self) -> str:
        if self.markdown:
            return self.markdown
        elif self.table_content:
            return "\n".join([" | ".join(row) for row in self.table_content])
        return self.content


@dataclass
class EquationBlock(ContentBlock):
    """公式块"""

    block_type: str = "equation"
    latex: str | None = None

    @property
    def equation_text(self) -> str:
        return self.latex or self.content


@dataclass
class MinerUOutput:
    """MinerU 统一输出格式"""

    blocks: list[ContentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    backend: str = ""  # 使用的后端 (pipeline 或 vlm-mlx-engine)
    page_images: list[dict[str, Any]] = field(default_factory=list)  # 每页图像信息

    def __post_init__(self):
        """按页面对块进行排序"""
        self.blocks.sort(key=lambda x: (x.page_idx, x.bbox.y0 if x.bbox else 0))

    @property
    def text_blocks(self) -> list[TextBlock]:
        """获取所有文本块"""
        return [block for block in self.blocks if isinstance(block, TextBlock)]

    @property
    def image_blocks(self) -> list[ImageBlock]:
        """获取所有图片块"""
        return [block for block in self.blocks if isinstance(block, ImageBlock)]

    @property
    def table_blocks(self) -> list[TableBlock]:
        """获取所有表格块"""
        return [block for block in self.blocks if isinstance(block, TableBlock)]

    @property
    def equation_blocks(self) -> list[EquationBlock]:
        """获取所有公式块"""
        return [block for block in self.blocks if isinstance(block, EquationBlock)]

    @property
    def plain_text(self) -> str:
        """获取纯文本"""
        texts = []
        for block in self.blocks:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ImageBlock) and block.caption:
                texts.append(" ".join(block.caption))
        return "\n\n".join(texts)

    @property
    def markdown(self) -> str:
        """获取 Markdown 格式"""
        md_parts = []
        for block in self.blocks:
            if isinstance(block, TextBlock):
                if block.text_level > 0:
                    md_parts.append("#" * block.text_level + " " + block.text)
                else:
                    md_parts.append(block.text)
            elif isinstance(block, ImageBlock):
                if block.img_path:
                    md_parts.append(f"![Image]({block.img_path})")
                if block.caption:
                    md_parts.append(" ".join(block.caption))
            elif isinstance(block, TableBlock) and block.markdown:
                md_parts.append(block.markdown)
            elif isinstance(block, EquationBlock):
                md_parts.append(f"${block.equation_text}$")
        return "\n\n".join(md_parts)

    def get_blocks_by_page(self, page_idx: int) -> list[ContentBlock]:
        """获取指定页面的所有块

        Args:
            page_idx: 页面索引（0-based，即第1页的page_idx为0）

        Returns:
            List[ContentBlock]: 指定页面的所有内容块

        Note:
            当使用page_range参数解析PDF时，page_idx会自动调整以反映实际页码。
            例如，如果解析第3-5页，那么第3页的块page_idx为2（0-based）。
        """
        return [block for block in self.blocks if block.page_idx == page_idx]

    def add_block(self, block: ContentBlock):
        """添加内容块"""
        self.blocks.append(block)

    def load_images(self, base_dir: Path | str | None = None):
        """加载所有图片的数据到内存

        Args:
            base_dir: 基础目录，如果图片路径是相对路径，则从此目录加载
        """
        if base_dir:
            base_dir = Path(base_dir)
        else:
            # 尝试从元数据中获取临时目录
            temp_dir = self.metadata.get("temp_dir")
            if temp_dir:
                base_dir = Path(temp_dir)

        for img_block in self.image_blocks:
            img_block.load_image_data(base_dir)

    def save_images(self, output_dir: Path | str):
        """保存所有图片到指定目录

        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_images = {}
        for i, img_block in enumerate(self.image_blocks):
            if img_block.img_path:
                # 使用原始文件名或生成新文件名
                img_name = Path(img_block.img_path).name
                if not img_name:
                    img_name = f"image_{i + 1}.jpg"

                # 确保文件名唯一
                output_file = output_path / img_name
                counter = 1
                while output_file.exists():
                    stem = Path(img_name).stem
                    suffix = Path(img_name).suffix
                    output_file = output_path / f"{stem}_{counter}{suffix}"
                    counter += 1

                saved_path = img_block.save_image(output_file)
                if saved_path:
                    saved_images[img_block.img_path] = saved_path

        return saved_images

    def save_all(self, output_dir: Path | str, save_text: bool = True, save_markdown: bool = True):
        """保存所有内容到目录

        Args:
            output_dir: 输出目录
            save_text: 是否保存纯文本
            save_markdown: 是否保存 Markdown
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 加载图片
        self.load_images()

        # 保存图片
        self.save_images(output_path / "images")

        # 保存文本
        if save_text:
            with open(output_path / "content.txt", "w", encoding="utf-8") as f:
                f.write(self.plain_text)

        # 保存 Markdown
        if save_markdown:
            with open(output_path / "content.md", "w", encoding="utf-8") as f:
                f.write(self.markdown)

        # 保存结构化数据
        import json

        structured_data = {"metadata": self.metadata, "pages": []}

        # 按页面组织数据
        max_page = max((b.page_idx for b in self.blocks), default=0)
        for page_idx in range(max_page + 1):
            page_blocks = self.get_blocks_by_page(page_idx)
            page_data = {"page_number": page_idx + 1, "blocks": []}

            for block in page_blocks:
                block_data = {"type": block.block_type, "content": block.content}

                if isinstance(block, TextBlock):
                    block_data.update({"text_level": block.text_level})
                elif isinstance(block, ImageBlock):
                    block_data.update(
                        {
                            "img_path": block.img_path,
                            "caption": block.caption,
                            "footnote": block.footnote,
                            "has_data": block.img_data is not None,
                        }
                    )

                if block.bbox:
                    block_data["bbox"] = [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1]

                page_data["blocks"].append(block_data)

            structured_data["pages"].append(page_data)

        with open(output_path / "structure.json", "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)


def _convert_content_list_to_blocks(content_list: list[dict]) -> MinerUOutput:
    """将原始 content_list 转换为标准化的 MinerUOutput

    Args:
        content_list: MinerU 原始输出的 content_list

    Returns:
        MinerUOutput: 标准化的输出对象
    """
    output = MinerUOutput()

    for item in content_list:
        # 提取边界框
        bbox = None
        if "bbox" in item and item["bbox"]:
            bbox_coords = item["bbox"]
            if len(bbox_coords) >= 4:
                bbox = BoundingBox(
                    x0=float(bbox_coords[0]),
                    y0=float(bbox_coords[1]),
                    x1=float(bbox_coords[2]),
                    y1=float(bbox_coords[3]),
                )

        # 提取页码
        page_idx = item.get("page_idx", 0)

        # 根据类型创建不同的块
        block_type = item.get("type", "text")

        if block_type in ["text", "header", "page_number", "footnote"]:
            # 创建文本块
            text_level = item.get("text_level", 0)
            if block_type == "header":
                text_level = max(text_level, 1)  # 标题至少是 level 1

            text_block = TextBlock(content=item.get("text", ""), bbox=bbox, page_idx=page_idx, text_level=text_level)
            output.add_block(text_block)

        elif block_type == "image":
            # 创建图片块
            img_path = item.get("img_path", "")
            image_block = ImageBlock(
                content=img_path,
                bbox=bbox,
                page_idx=page_idx,
                img_path=img_path,
                caption=item.get("image_caption", []),
                footnote=item.get("image_footnote", []),
            )
            output.add_block(image_block)

        elif block_type == "table":
            # 创建表格块
            table_block = TableBlock(
                content=item.get("text", ""), bbox=bbox, page_idx=page_idx, markdown=item.get("table_caption", "")
            )
            output.add_block(table_block)

        elif block_type == "equation":
            # 创建公式块
            equation_block = EquationBlock(
                content=item.get("text", ""), bbox=bbox, page_idx=page_idx, latex=item.get("latex", "")
            )
            output.add_block(equation_block)

    return output


def _convert_middle_json_to_blocks(middle_json: dict) -> MinerUOutput:
    """将 middle_json 转换为标准化的 MinerUOutput

    Args:
        middle_json: MinerU 原始输出的 middle_json

    Returns:
        MinerUOutput: 标准化的输出对象
    """
    output = MinerUOutput()

    # 提取页面信息
    pages = middle_json.get("pages", [])

    for page_idx, page_data in enumerate(pages):
        # 处理每个块
        blocks = page_data.get("blocks", [])

        for block_data in blocks:
            # 获取边界框
            bbox = None
            if "bbox" in block_data and block_data["bbox"]:
                bbox_coords = block_data["bbox"]
                if len(bbox_coords) >= 4:
                    bbox = BoundingBox(
                        x0=float(bbox_coords[0]),
                        y0=float(bbox_coords[1]),
                        x1=float(bbox_coords[2]),
                        y1=float(bbox_coords[3]),
                    )

            # 获取块类型
            block_type = block_data.get("type", "text")

            # 根据类型创建不同的块
            if block_type in ["text", "title", "subtitle"]:
                text_level = block_data.get("level", 0)
                if block_type in ["title", "subtitle"]:
                    text_level = text_level + 1

                text_block = TextBlock(
                    content=block_data.get("text", ""), bbox=bbox, page_idx=page_idx, text_level=text_level
                )
                output.add_block(text_block)

            elif block_type == "image":
                img_path = block_data.get("path", "")
                image_block = ImageBlock(
                    content=img_path,
                    bbox=bbox,
                    page_idx=page_idx,
                    img_path=img_path,
                    caption=block_data.get("captions", []),
                    footnote=block_data.get("footnotes", []),
                )
                output.add_block(image_block)

            elif block_type == "table":
                table_block = TableBlock(
                    content=block_data.get("text", ""),
                    bbox=bbox,
                    page_idx=page_idx,
                    markdown=block_data.get("markdown", ""),
                )
                output.add_block(table_block)

            elif block_type == "equation":
                equation_block = EquationBlock(
                    content=block_data.get("text", ""), bbox=bbox, page_idx=page_idx, latex=block_data.get("latex", "")
                )
                output.add_block(equation_block)

    return output


def parse_pipeline_model_json(model_json: list[dict], page_info_list: list[tuple[Path, int, int]], page_offset: int = 0) -> MinerUOutput:
    """解析pipeline后端的model.json输出

    Args:
        model_json: pipeline后端的model.json内容
        page_info_list: 每页的图像信息列表 (path, width, height)
        page_offset: 页面偏移量（当使用page_range时需要调整page_idx）

    Returns:
        MinerUOutput: 标准化输出
    """
    from pydantic import BaseModel
    from enum import IntEnum

    # 定义CategoryType枚举
    class CategoryType(IntEnum):
        """内容类别枚举"""

        title = 0
        plain_text = 1
        abandon = 2
        figure = 3
        figure_caption = 4
        table = 5
        table_caption = 6
        table_footnote = 7
        isolate_formula = 8
        formula_caption = 9
        embedding = 13
        isolated = 14
        text = 15

    output = MinerUOutput()
    output.backend = "pipeline"

    # 添加页面图像信息
    for page_idx, (img_path, width, height) in enumerate(page_info_list):
        # 调整page_idx以反映实际的页面编号（考虑page_offset）
        actual_page_idx = page_idx + page_offset
        output.page_images.append({"page_idx": actual_page_idx, "image_path": str(img_path), "width": width, "height": height})

    # 映射CategoryType到block_type
    category_map = {
        CategoryType.title: "title",
        CategoryType.plain_text: "text",
        CategoryType.figure: "image",
        CategoryType.figure_caption: "image_caption",
        CategoryType.table: "table",
        CategoryType.table_caption: "table_caption",
        CategoryType.table_footnote: "table_footnote",
        CategoryType.isolate_formula: "equation",
        CategoryType.formula_caption: "equation_caption",
        CategoryType.embedding: "equation",
        CategoryType.isolated: "equation",
        CategoryType.text: "text",
    }

    for page_idx, page_data in enumerate(model_json):
        # 调整page_idx以反映实际的页面编号（考虑page_offset）
        actual_page_idx = page_idx + page_offset

        # 处理每个检测到的对象
        for obj in page_data.get("layout_dets", []):
            category_id = obj.get("category_id", 1)
            poly = obj.get("poly", [])
            latex = obj.get("latex")
            html = obj.get("html")

            # 将四边形坐标转换为bbox
            if len(poly) >= 8:
                x0, y0 = poly[0], poly[1]
                x1, y1 = poly[4], poly[5]
                # 使用绝对坐标（pipeline后端已经是像素坐标）
                bbox = BoundingBox(x0, y0, x1, y1)
            else:
                bbox = None

            # 获取内容类型
            block_type = category_map.get(CategoryType(category_id), "text")

            # 创建相应的内容块
            if block_type in ["text", "title"]:
                text_level = 1 if block_type == "title" else 0
                content = latex or html or ""
                text_block = TextBlock(content=content, bbox=bbox, page_idx=actual_page_idx, text_level=text_level)
                output.add_block(text_block)

            elif block_type == "image":
                image_block = ImageBlock(
                    content="",  # 没有直接的图像内容
                    bbox=bbox,
                    page_idx=actual_page_idx,
                )
                output.add_block(image_block)

            elif block_type == "table":
                table_block = TableBlock(content=html or "", bbox=bbox, page_idx=actual_page_idx)
                output.add_block(table_block)

            elif block_type == "equation":
                equation_block = EquationBlock(content=latex or "", bbox=bbox, page_idx=actual_page_idx, latex=latex)
                output.add_block(equation_block)

    return output


def parse_vlm_model_json(model_json: list[dict], page_info_list: list[tuple[Path, int, int]], page_offset: int = 0) -> MinerUOutput:
    """解析VLM后端的model.json输出

    Args:
        model_json: VLM后端的model.json内容
        page_info_list: 每页的图像信息列表 (path, width, height)
        page_offset: 页面偏移量（当使用page_range时需要调整page_idx）

    Returns:
        MinerUOutput: 标准化输出
    """
    output = MinerUOutput()
    output.backend = "vlm-mlx-engine"

    # 添加页面图像信息
    for page_idx, (img_path, width, height) in enumerate(page_info_list):
        # 调整page_idx以反映实际的页面编号（考虑page_offset）
        actual_page_idx = page_idx + page_offset
        output.page_images.append({"page_idx": actual_page_idx, "image_path": str(img_path), "width": width, "height": height})

    # VLM后端的类型映射
    vlm_type_map = {
        "text": "text",
        "title": "title",
        "subtitle": "title",
        "equation": "equation",
        "image": "image",
        "image_caption": "image_caption",
        "image_footnote": "image_footnote",
        "table": "table",
        "table_caption": "table_caption",
        "table_footnote": "table_footnote",
        "phonetic": "text",
        "code": "text",
        "code_caption": "text",
        "ref_text": "text",
        "algorithm": "text",
        "list": "text",
        "header": "text",  # 通常忽略
        "footer": "text",  # 通常忽略
        "page_number": "text",  # 通常忽略
        "aside_text": "text",
        "page_footnote": "text",
    }

    # 临时存储所有 blocks，用于后续处理 ImageCaption 的合并
    temp_blocks = []

    for page_idx, page_blocks in enumerate(model_json):
        # 调整page_idx以反映实际的页面编号（考虑page_offset）
        actual_page_idx = page_idx + page_offset

        # 获取页面尺寸
        if page_idx < len(page_info_list):
            _, page_width, page_height = page_info_list[page_idx]
        else:
            page_width, page_height = 0, 0

        for block_data in page_blocks:
            block_type = block_data.get("type", "text")
            content = block_data.get("content", "")
            bbox_data = block_data.get("bbox", [])

            # 处理边界框（VLM后端是相对坐标0-1）
            bbox = None
            if len(bbox_data) >= 4 and page_width > 0 and page_height > 0:
                x0, y0, x1, y1 = bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]
                # VLM后端的坐标是相对坐标，需要转换为绝对坐标
                bbox = BoundingBox(x0, y0, x1, y1, page_width, page_height)

            # 映射类型
            mapped_type = vlm_type_map.get(block_type, "text")

            # 创建相应的内容块，但先不添加到 output 中
            if mapped_type in ["text", "title"]:
                text_level = 1 if mapped_type == "title" else 0
                text_block = TextBlock(content=content, bbox=bbox, page_idx=actual_page_idx, text_level=text_level)
                temp_blocks.append(text_block)

            elif mapped_type == "image":
                image_block = ImageBlock(content="", bbox=bbox, page_idx=actual_page_idx)
                temp_blocks.append(image_block)

            elif mapped_type == "image_caption":
                # 暂时创建 ImageBlock，后续会处理合并或转换
                image_block = ImageBlock(content=content, bbox=bbox, page_idx=actual_page_idx, caption=[content])
                temp_blocks.append(("image_caption", image_block))

            elif mapped_type == "table":
                table_block = TableBlock(content=content, bbox=bbox, page_idx=actual_page_idx)
                temp_blocks.append(table_block)

            elif mapped_type == "equation":
                equation_block = EquationBlock(content=content, bbox=bbox, page_idx=actual_page_idx, latex=content)
                temp_blocks.append(equation_block)

    # 处理 ImageCaption 的合并逻辑
    i = 0
    while i < len(temp_blocks):
        item = temp_blocks[i]

        if isinstance(item, tuple) and item[0] == "image_caption":
            caption_block = item[1]
            content = caption_block.content
            bbox = caption_block.bbox
            page_idx = caption_block.page_idx

            # 向前查找 Image（-1位置）
            found_image = False

            # 检查前一个 block
            if i > 0 and isinstance(temp_blocks[i-1], ImageBlock):
                prev_block = temp_blocks[i-1]
                if isinstance(prev_block, ImageBlock) and prev_block.content == "":
                    # 找到了 Image，合并 caption
                    prev_block.caption.extend([content])
                    found_image = True

            # 如果前面没找到，向后查找 Image（+1位置）
            if not found_image and i < len(temp_blocks) - 1:
                if isinstance(temp_blocks[i+1], ImageBlock):
                    next_block = temp_blocks[i+1]
                    if isinstance(next_block, ImageBlock) and next_block.content == "":
                        # 找到了 Image，合并 caption
                        next_block.caption.extend([content])
                        found_image = True

            # 如果都没找到 Image，将 ImageCaption 转为 TextBlock
            if not found_image:
                text_block = TextBlock(
                    content=content,
                    bbox=bbox,
                    page_idx=page_idx,
                    text_level=0
                )
                # 用 TextBlock 替换当前的 ImageCaption
                temp_blocks[i] = text_block
            else:
                # 如果找到了 Image 并合并了，移除这个 ImageCaption
                temp_blocks.pop(i)
                i -= 1  # 因为移除了一个元素，索引需要减1

        i += 1

    # 处理框重叠去重
    temp_blocks = remove_overlapping_blocks(temp_blocks)

    # 将处理后的 blocks 添加到 output 中
    for block in temp_blocks:
        if isinstance(block, ContentBlock):
            output.add_block(block)

    return output


def remove_overlapping_blocks(blocks: list) -> list:
    """移除重叠的块，根据以下规则：
    1. 如果某个框的中心点位于其他框内部，检测是否有重叠
    2. 如果一个有content，一个没有，则去掉没有content的
    3. 如果两个都有content或两个都没有content，则去除内部的框

    Args:
        blocks: 内容块列表

    Returns:
        List[ContentBlock]: 处理后的内容块列表
    """
    if not blocks:
        return blocks

    # 只处理有 bbox 的块
    blocks_with_bbox = [block for block in blocks if hasattr(block, 'bbox') and block.bbox is not None]
    blocks_to_remove = set()

    for i, block1 in enumerate(blocks_with_bbox):
        if i in blocks_to_remove:
            continue

        # 计算 block1 的中心点
        center1_x = (block1.bbox.x0 + block1.bbox.x1) / 2
        center1_y = (block1.bbox.y0 + block1.bbox.y1) / 2

        for j, block2 in enumerate(blocks_with_bbox):
            if i == j or j in blocks_to_remove:
                continue

            # 检查 block1 的中心点是否在 block2 内部
            if (block2.bbox.x0 <= center1_x <= block2.bbox.x1 and
                block2.bbox.y0 <= center1_y <= block2.bbox.y1):

                # 检查 block2 的中心点是否也在 block1 内部（互相包含）
                center2_x = (block2.bbox.x0 + block2.bbox.x1) / 2
                center2_y = (block2.bbox.y0 + block2.bbox.y1) / 2

                center2_in_block1 = (block1.bbox.x0 <= center2_x <= block1.bbox.x1 and
                                    block1.bbox.y0 <= center2_y <= block1.bbox.y1)

                # 获取 content 内容
                content1 = getattr(block1, 'content', '') or ''
                content2 = getattr(block2, 'content', '') or ''

                # 对于 ImageBlock，还要考虑 caption 和 footnote
                if hasattr(block1, 'caption') and block1.caption:
                    content1 = content1 + ' '.join(block1.caption)
                if hasattr(block1, 'footnote') and block1.footnote:
                    content1 = content1 + ' '.join(block1.footnote)
                if hasattr(block2, 'caption') and block2.caption:
                    content2 = content2 + ' '.join(block2.caption)
                if hasattr(block2, 'footnote') and block2.footnote:
                    content2 = content2 + ' '.join(block2.footnote)

                has_content1 = bool(content1.strip())
                has_content2 = bool(content2.strip())

                # 判断哪个块应该被移除
                remove_index = None

                if not center2_in_block1:
                    # 只有 block1 的中心在 block2 内，block1 是内部的
                    if has_content1 and not has_content2:
                        # block1 有内容，block2 没有内容 -> 移除 block2
                        remove_index = j
                    elif has_content2 and not has_content1:
                        # block2 有内容，block1 没有内容 -> 移除 block1
                        remove_index = i
                    else:
                        # 都有内容或都没有内容 -> 移除内部的（block1）
                        remove_index = i
                else:
                    # 互相包含，选择面积较小的移除
                    area1 = (block1.bbox.x1 - block1.bbox.x0) * (block1.bbox.y1 - block1.bbox.y0)
                    area2 = (block2.bbox.x1 - block2.bbox.x0) * (block2.bbox.y1 - block2.bbox.y0)

                    if area1 < area2:
                        remove_index = i
                    else:
                        remove_index = j

                    # 如果一个有内容一个没有，优先保留有内容的
                    if has_content1 and not has_content2 and remove_index == i:
                        remove_index = j
                    elif has_content2 and not has_content1 and remove_index == j:
                        remove_index = i

                if remove_index is not None:
                    blocks_to_remove.add(remove_index)

    # 构建结果列表
    result = []
    for i, block in enumerate(blocks):
        if isinstance(block, tuple):
            # 这种情况不应该出现，但为了安全起见
            continue
        if i not in blocks_to_remove:
            result.append(block)

    return result


def parse_pdf_to_content_list(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    formula_enable: bool = True,
    table_enable: bool = True,
    **kwargs,
) -> list[dict]:
    """
    解析 PDF 文档并返回内容列表

    Args:
        pdf_path: PDF 文件路径或 Path 对象
        page_range: 页面范围元组 (start_page, end_page)，None 表示全部页面
        output_dir: 输出目录，None 表示使用临时目录
        lang: 文档语言，默认为 "ch"（中文）
        backend: 后端引擎，默认为 "vlm-mlx-engine"
        formula_enable: 是否启用公式识别
        table_enable: 是否启用表格识别
        **kwargs: 其他传递给 do_parse 的参数

    Returns:
        List[Dict]: 内容列表，每个元素包含类型、文本、边界框等信息
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 创建输出目录
    if output_dir is None:
        # 使用临时目录
        temp_dir = tempfile.mkdtemp(prefix="mineru_")
        output_dir = Path(temp_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 设置环境变量以尝试使用 MPS（如果可用）
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 将 PDF 转换为字节数据
    pdf_bytes = read_fn(pdf_path)
    pdf_file_name = pdf_path.stem

    # 设置页面范围
    start_page_id = 0
    end_page_id = None
    if page_range is not None:
        start_page_id, end_page_id = page_range
        # 转换为 0-based 索引
        if start_page_id > 0:
            start_page_id -= 1
        if end_page_id is not None and end_page_id > 0:
            end_page_id -= 1

    try:
        # 调用 do_parse 进行解析
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[lang],
            backend=backend,
            parse_method="vlm",
            p_formula_enable=formula_enable,
            p_table_enable=table_enable,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **kwargs,
        )

        # 读取生成的 content_list.json 文件
        content_list_path = output_dir / pdf_file_name / "vlm" / f"{pdf_file_name}_content_list.json"

        if not content_list_path.exists():
            raise FileNotFoundError(f"内容列表文件未生成: {content_list_path}")

        with open(content_list_path, encoding="utf-8") as f:
            content_list = json.load(f)

        return content_list

    except Exception as e:
        raise RuntimeError(f"解析 PDF 失败: {e}") from e

    finally:
        # 如果使用临时目录，可以选择删除或保留
        # 这里保留临时目录以便调试
        pass


def parse_pdf_to_mineru_output(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    formula_enable: bool = True,
    table_enable: bool = True,
    dpi: int = 200,
    **kwargs,
) -> MinerUOutput:
    """
    解析 PDF 文档并返回标准化的 MinerUOutput 对象

    该函数会自动将PDF转换为DPI=200的PNG图像，并根据使用的后端
    (pipeline 或 vlm-mlx-engine) 正确处理坐标系统。

    Args:
        pdf_path: PDF 文件路径或 Path 对象
        page_range: 页面范围元组 (start_page, end_page)，None 表示全部页面
        output_dir: 输出目录，None 表示使用临时目录
        lang: 文档语言，默认为 "ch"（中文）
        backend: 后端引擎，支持 "pipeline" 或 "vlm-mlx-engine"
        formula_enable: 是否启用公式识别
        table_enable: 是否启用表格识别
        dpi: PDF转PNG的DPI，默认200
        **kwargs: 其他传递给 do_parse 的参数

    Returns:
        MinerUOutput: 标准化的输出对象，包含内容块和页面图像信息
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 创建输出目录
    if output_dir is None:
        # 使用临时目录
        temp_dir = tempfile.mkdtemp(prefix="mineru_")
        output_dir = Path(temp_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 将PDF转换为PNG图像（DPI=200）
    print(f"正在将PDF转换为PNG图像（DPI={dpi}）...")
    png_output_dir = output_dir / "images"

    # 如果指定了页面范围，转换为0-based索引用于PNG转换
    png_page_range = None
    if page_range is not None:
        start_page, end_page = page_range
        # 转换为0-based索引
        start_idx = max(0, start_page - 1) if start_page > 0 else 0
        end_idx = end_page - 1 if end_page > 0 else 0
        png_page_range = (start_idx, end_idx)

    # 使用页面范围转换PNG
    page_info_list = convert_pdf_to_png(pdf_path, png_output_dir, dpi=dpi, page_range=png_page_range)

    # 设置环境变量以尝试使用 MPS（如果可用）
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 将 PDF 转换为字节数据
    pdf_bytes = read_fn(pdf_path)
    pdf_file_name = pdf_path.stem

    # 设置页面范围
    start_page_id = 0
    end_page_id = None
    if page_range is not None:
        start_page_id, end_page_id = page_range
        # 转换为 0-based 索引
        if start_page_id > 0:
            start_page_id -= 1
        if end_page_id is not None and end_page_id > 0:
            end_page_id -= 1

    try:
        # 调用 do_parse 进行解析，启用 model_output
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[lang],
            backend=backend,
            parse_method="vlm",
            p_formula_enable=formula_enable,
            p_table_enable=table_enable,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,  # 不需要middle_json
            f_dump_model_output=True,  # 启用model.json输出
            f_dump_orig_pdf=False,
            f_dump_content_list=False,  # 不需要content_list
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **kwargs,
        )

        # 读取生成的 model.json 文件
        # 根据后端不同，文件路径可能不同
        model_json_path = None
        model_json = None

        # 尝试多个可能的路径
        possible_paths = [
            output_dir / pdf_file_name / "model.json",  # pipeline直接输出
            output_dir / pdf_file_name / "vlm" / f"{pdf_file_name}_model.json",  # VLM输出
            output_dir / pdf_file_name / "vlm" / "model.json",  # 另一种可能的路径
        ]

        for path in possible_paths:
            if path.exists():
                model_json_path = path
                break

        if not model_json_path:
            raise FileNotFoundError(f"模型输出文件未生成，尝试的路径: {[str(p) for p in possible_paths]}")

        with open(model_json_path, encoding="utf-8") as f:
            model_json = json.load(f)

        # 计算页面偏移量（当指定了page_range时）
        page_offset = 0
        if page_range is not None:
            page_offset = max(0, page_range[0] - 1) if page_range[0] > 0 else 0

        # 检测实际的数据格式来决定使用哪种解析器
        # 而不是完全依赖backend参数
        if model_json and isinstance(model_json, list) and model_json:
            first_item = model_json[0]
            if isinstance(first_item, dict) and 'layout_dets' in first_item:
                # Pipeline格式
                print("检测到pipeline格式，使用pipeline解析器...")
                return parse_pipeline_model_json(model_json, page_info_list, page_offset=page_offset)
            elif isinstance(first_item, list):
                # VLM格式
                print("检测到VLM格式，使用VLM解析器...")
                return parse_vlm_model_json(model_json, page_info_list, page_offset=page_offset)

        # 如果格式检测失败，根据backend参数决定
        if backend == "pipeline":
            print("使用pipeline后端解析结果...")
            return parse_pipeline_model_json(model_json, page_info_list, page_offset=page_offset)
        else:
            print("使用VLM后端解析结果...")
            return parse_vlm_model_json(model_json, page_info_list, page_offset=page_offset)

    except Exception as e:
        raise RuntimeError(f"解析 PDF 失败: {e}") from e

    finally:
        # 如果使用临时目录，可以选择删除或保留
        # 这里保留临时目录以便调试
        pass


def parse_pdf_to_middle_json(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    formula_enable: bool = True,
    table_enable: bool = True,
    **kwargs,
) -> dict:
    """
    解析 PDF 文档并返回中间 JSON 结果

    Args:
        pdf_path: PDF 文件路径或 Path 对象
        page_range: 页面范围元组 (start_page, end_page)，None 表示全部页面
        output_dir: 输出目录，None 表示使用临时目录
        lang: 文档语言，默认为 "ch"（中文）
        backend: 后端引擎，默认为 "vlm-mlx-engine"
        formula_enable: 是否启用公式识别
        table_enable: 是否启用表格识别
        **kwargs: 其他传递给 do_parse 的参数

    Returns:
        Dict: 中间 JSON 结果，包含详细的解析信息
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 创建输出目录
    if output_dir is None:
        # 使用临时目录
        temp_dir = tempfile.mkdtemp(prefix="mineru_")
        output_dir = Path(temp_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 设置环境变量以尝试使用 MPS（如果可用）
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 将 PDF 转换为字节数据
    pdf_bytes = read_fn(pdf_path)
    pdf_file_name = pdf_path.stem

    # 设置页面范围
    start_page_id = 0
    end_page_id = None
    if page_range is not None:
        start_page_id, end_page_id = page_range
        # 转换为 0-based 索引
        if start_page_id > 0:
            start_page_id -= 1
        if end_page_id is not None and end_page_id > 0:
            end_page_id -= 1

    try:
        # 调用 do_parse 进行解析
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[lang],
            backend=backend,
            parse_method="vlm",
            p_formula_enable=formula_enable,
            p_table_enable=table_enable,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **kwargs,
        )

        # 读取生成的 middle_json 文件
        middle_json_path = output_dir / pdf_file_name / "vlm" / f"{pdf_file_name}_middle.json"

        if not middle_json_path.exists():
            raise FileNotFoundError(f"中间 JSON 文件未生成: {middle_json_path}")

        with open(middle_json_path, encoding="utf-8") as f:
            middle_json = json.load(f)

        return middle_json

    except Exception as e:
        raise RuntimeError(f"解析 PDF 失败: {e}") from e

    finally:
        # 如果使用临时目录，可以选择删除或保留
        # 这里保留临时目录以便调试
        pass


def extract_text_from_content_list(content_list: list[dict]) -> str:
    """
    从内容列表中提取纯文本

    Args:
        content_list: MinerU 解析的内容列表

    Returns:
        str: 提取的纯文本
    """
    text_parts = []

    for item in content_list:
        if item.get("type") == "text":
            text_parts.append(item.get("text", ""))
        elif item.get("type") == "header":
            text_parts.append(item.get("text", ""))
        elif item.get("type") == "page_number":
            # 通常不需要页码
            continue
        elif item.get("type") == "image":
            # 可以选择添加图片说明
            captions = item.get("image_caption", [])
            if captions:
                text_parts.append(" ".join(captions))

    return "\n\n".join(text_parts)


def extract_text_blocks_from_content_list(content_list: list[dict]) -> list[dict]:
    """
    从内容列表中提取文本块

    Args:
        content_list: MinerU 解析的内容列表

    Returns:
        List[Dict]: 文本块列表，每个元素包含文本、类型和位置信息
    """
    text_blocks = []

    for item in content_list:
        if item.get("type") in ["text", "header"]:
            text_blocks.append(
                {
                    "text": item.get("text", ""),
                    "type": item.get("type", ""),
                    "bbox": item.get("bbox", []),
                    "page_idx": item.get("page_idx", 0),
                    "text_level": item.get("text_level", 0),
                }
            )

    return text_blocks


def extract_images_from_content_list(content_list: list[dict], output_dir: Path | None = None) -> list[dict]:
    """
    从内容列表中提取图片信息

    Args:
        content_list: MinerU 解析的内容列表
        output_dir: 输出目录，None 表示使用临时目录

    Returns:
        List[Dict]: 图片信息列表，每个元素包含图片路径、说明和位置信息
    """
    image_list = []

    for item in content_list:
        if item.get("type") == "image":
            image_info = {
                "img_path": item.get("img_path", ""),
                "image_caption": item.get("image_caption", []),
                "image_footnote": item.get("image_footnote", []),
                "bbox": item.get("bbox", []),
                "page_idx": item.get("page_idx", 0),
            }

            # 如果提供了输出目录，将相对路径转换为绝对路径
            if output_dir and image_info["img_path"]:
                img_path = output_dir / image_info["img_path"]
                if img_path.exists():
                    image_info["absolute_path"] = str(img_path)

            image_list.append(image_info)

    return image_list


def save_content_list_to_json(content_list: list[dict], output_path: str | Path) -> None:
    """
    将内容列表保存到 JSON 文件

    Args:
        content_list: MinerU 解析的内容列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(content_list, f, ensure_ascii=False, indent=2)


def save_middle_json_to_json(middle_json: dict, output_path: str | Path) -> None:
    """
    将中间 JSON 保存到文件

    Args:
        middle_json: MinerU 解析的中间 JSON
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(middle_json, f, ensure_ascii=False, indent=2)


# 便捷函数：直接从 PDF 文件提取文本
def parse_pdf(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    formula_enable: bool = True,
    table_enable: bool = True,
    return_format: str = "content_list",  # "content_list" or "middle_json"
    **kwargs,
) -> MinerUOutput:
    """统一的 PDF 解析函数，支持不同后端并返回标准化的输出格式

    Args:
        pdf_path: PDF 文件路径或 Path 对象
        page_range: 页面范围元组 (start_page, end_page)，None 表示全部页面
        output_dir: 输出目录，None 表示使用临时目录
        lang: 文档语言，默认为 "ch"（中文）
        backend: 后端引擎 ("vlm-mlx-engine" 或 "pipeline")
        formula_enable: 是否启用公式识别
        table_enable: 是否启用表格识别
        return_format: 返回格式 ("content_list" 或 "middle_json")
        **kwargs: 其他传递给 do_parse 的参数

    Returns:
        MinerUOutput: 标准化的输出对象，包含所有解析的内容

    Example:
        >>> # 解析 PDF 并获取所有文本块
        >>> output = parse_pdf("document.pdf")
        >>> for text_block in output.text_blocks:
        ...     print(text_block.text)
        >>>
        >>> # 获取所有图片
        >>> for img_block in output.image_blocks:
        ...     print(f"Image: {img_block.img_path}")
        ...     print(f"Caption: {img_block.caption}")
        >>>
        >>> # 获取纯文本
        >>> text = output.plain_text
        >>>
        >>> # 获取 Markdown 格式
        >>> md = output.markdown
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 根据返回格式选择解析方法
    if return_format == "content_list":
        # 使用 content_list 格式解析
        temp_dir = None
        if output_dir is None:
            # 使用临时目录
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix="mineru_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
        print(output_dir)
        content_list = parse_pdf_to_content_list(
            pdf_path=pdf_path,
            page_range=page_range,
            output_dir=output_dir,
            lang=lang,
            backend=backend,
            formula_enable=formula_enable,
            table_enable=table_enable,
            **kwargs,
        )

        # 转换为标准化输出
        output = _convert_content_list_to_blocks(content_list)

        # 添加元数据，包括图片的路径信息
        output.metadata.update(
            {
                "pdf_path": str(pdf_path),
                "page_range": page_range,
                "backend": backend,
                "lang": lang,
                "temp_dir": temp_dir,
                "output_dir": str(output_dir),
                "image_base_dir": str(output_dir / pdf_path.stem / "vlm"),
            }
        )

        # 预加载图片数据
        output.load_images(output.metadata["image_base_dir"])

        # 保存图片到 output_dir 并更新路径
        if output.image_blocks:
            # 创建 images 目录
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # 保存图片并获取新路径映射
            saved_images = {}
            for i, img_block in enumerate(output.image_blocks):
                if img_block.img_data:
                    # 生成图片文件名
                    img_name = Path(img_block.img_path).name if img_block.img_path else f"image_{i + 1:03d}.jpg"
                    if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                        img_name += ".jpg"

                    # 确保文件名唯一
                    img_file_path = images_dir / img_name
                    counter = 1
                    while img_file_path.exists():
                        stem = Path(img_name).stem
                        suffix = Path(img_name).suffix
                        img_file_path = images_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                    # 保存图片
                    img_file_path.write_bytes(img_block.img_data)

                    # 记录新旧路径映射
                    old_path = img_block.img_path
                    new_relative_path = f"images/{img_file_path.name}"
                    img_block.img_path = new_relative_path
                    if old_path:
                        saved_images[old_path] = new_relative_path

        return output

    elif return_format == "middle_json":
        # 使用 middle_json 格式解析
        temp_dir = None
        if output_dir is None:
            # 使用临时目录
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix="mineru_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)

        middle_json = parse_pdf_to_middle_json(
            pdf_path=pdf_path,
            page_range=page_range,
            output_dir=output_dir,
            lang=lang,
            backend=backend,
            formula_enable=formula_enable,
            table_enable=table_enable,
            **kwargs,
        )

        # 转换为标准化输出
        output = _convert_middle_json_to_blocks(middle_json)

        # 添加元数据
        output.metadata.update(
            {
                "pdf_path": str(pdf_path),
                "page_range": page_range,
                "backend": backend,
                "lang": lang,
                "temp_dir": temp_dir,
                "output_dir": str(output_dir),
                "image_base_dir": str(output_dir / pdf_path.stem / "vlm"),
            }
        )

        # 预加载图片数据
        output.load_images(output.metadata["image_base_dir"])

        # 保存图片到 output_dir 并更新路径
        if output.image_blocks:
            # 创建 images 目录
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # 保存图片并获取新路径映射
            saved_images = {}
            for i, img_block in enumerate(output.image_blocks):
                if img_block.img_data:
                    # 生成图片文件名
                    img_name = Path(img_block.img_path).name if img_block.img_path else f"image_{i + 1:03d}.jpg"
                    if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                        img_name += ".jpg"

                    # 确保文件名唯一
                    img_file_path = images_dir / img_name
                    counter = 1
                    while img_file_path.exists():
                        stem = Path(img_name).stem
                        suffix = Path(img_name).suffix
                        img_file_path = images_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                    # 保存图片
                    img_file_path.write_bytes(img_block.img_data)

                    # 记录新旧路径映射
                    old_path = img_block.img_path
                    new_relative_path = f"images/{img_file_path.name}"
                    img_block.img_path = new_relative_path
                    if old_path:
                        saved_images[old_path] = new_relative_path

        return output

    else:
        raise ValueError(f"不支持的返回格式: {return_format}")


def parse_pdf_simple(
    pdf_path: str | Path,
    backend: str = "vlm-mlx-engine",
    **kwargs,
) -> MinerUOutput:
    """简化的 PDF 解析函数，使用默认参数解析 PDF

    Args:
        pdf_path: PDF 文件路径
        backend: 后端引擎
        **kwargs: 其他参数

    Returns:
        MinerUOutput: 标准化的输出对象
    """
    return parse_pdf(
        pdf_path=pdf_path,
        backend=backend,
        **kwargs,
    )


def extract_text_from_pdf(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    **kwargs,
) -> str:
    """
    直接从 PDF 文件提取文本

    Args:
        pdf_path: PDF 文件路径或 Path 对象
        page_range: 页面范围元组 (start_page, end_page)，None 表示全部页面
        output_dir: 输出目录，None 表示使用临时目录
        lang: 文档语言，默认为 "ch"（中文）
        backend: 后端引擎，默认为 "vlm-mlx-engine"
        **kwargs: 其他传递给 do_parse 的参数

    Returns:
        str: 提取的文本内容
    """
    # 使用新的标准化解析函数
    output = parse_pdf_simple(
        pdf_path=pdf_path,
        page_range=page_range,
        output_dir=output_dir,
        lang=lang,
        backend=backend,
        **kwargs,
    )

    return output.plain_text


# 便捷函数：直接从 PDF 文件提取内容列表
def extract_content_list_from_pdf(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    save_result: bool = False,
    save_path: str | Path | None = None,
    return_standardized: bool = False,
    **kwargs,
) -> list[dict] | MinerUOutput:
    """
    直接从 PDF 文件提取内容列表

    Args:
        pdf_path: PDF 文件路径或 Path 对象
        page_range: 页面范围元组 (start_page, end_page)，None 表示全部页面
        output_dir: 输出目录，None 表示使用临时目录
        lang: 文档语言，默认为 "ch"（中文）
        backend: 后端引擎，默认为 "vlm-mlx-engine"
        save_result: 是否保存结果到文件
        save_path: 结果保存路径，None 表示使用默认路径
        return_standardized: 是否返回标准化的 MinerUOutput 对象
        **kwargs: 其他传递给 do_parse 的参数

    Returns:
        Union[List[Dict], MinerUOutput]: 内容列表或标准化的输出对象
    """
    if return_standardized:
        # 返回标准化的输出
        output = parse_pdf_simple(
            pdf_path=pdf_path,
            page_range=page_range,
            output_dir=output_dir,
            lang=lang,
            backend=backend,
            **kwargs,
        )

        if save_result:
            # 如果需要保存，转换为原始格式并保存
            content_list = []
            for block in output.blocks:
                item = {
                    "type": block.block_type,
                    "text": block.content,
                    "page_idx": block.page_idx,
                }
                if block.bbox:
                    item["bbox"] = [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1]

                # 根据块类型添加特定字段
                if isinstance(block, TextBlock):
                    item["text_level"] = block.text_level
                elif isinstance(block, ImageBlock):
                    item["img_path"] = block.img_path
                    item["image_caption"] = block.caption
                    item["image_footnote"] = block.footnote
                elif isinstance(block, TableBlock) and block.markdown:
                    item["table_caption"] = block.markdown
                elif isinstance(block, EquationBlock) and block.latex:
                    item["latex"] = block.latex

                content_list.append(item)

            if save_path is None:
                pdf_path = Path(pdf_path)
                save_path = pdf_path.parent / f"{pdf_path.stem}_content_list.json"

            save_content_list_to_json(content_list, save_path)

        return output
    else:
        # 返回原始格式
        content_list = parse_pdf_to_content_list(
            pdf_path=pdf_path, page_range=page_range, output_dir=output_dir, lang=lang, backend=backend, **kwargs
        )

        if save_result:
            if save_path is None:
                pdf_path = Path(pdf_path)
                save_path = pdf_path.parent / f"{pdf_path.stem}_content_list.json"

            save_content_list_to_json(content_list, save_path)

        return content_list


def visualize_blocks_on_image(mineru_output: MinerUOutput,
                             output_dir: str | Path,
                             draw_text: bool = True,
                             draw_image: bool = True,
                             draw_table: bool = True,
                             draw_equation: bool = True,
                             thickness: int = 2) -> list[str]:
    """在图像上可视化所有检测到的块

    Args:
        mineru_output: MinerUOutput对象，包含所有检测到的块
        output_dir: 输出目录，保存可视化结果
        draw_text: 是否绘制文本块
        draw_image: 是否绘制图像块
        draw_table: 是否绘制表格块
        draw_equation: 是否绘制公式块
        thickness: 边框线宽

    Returns:
        List[str]: 可视化图像路径列表
    """
    if 'cv2' not in globals() or 'np' not in globals():
        print("错误：OpenCV未安装，无法进行可视化")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义不同类型块的颜色 (BGR格式)
    colors = {
        'text': (0, 255, 0),      # 绿色
        'title': (0, 0, 255),     # 红色
        'image': (255, 0, 0),     # 蓝色
        'image_caption': (255, 0, 255),  # 紫色
        'table': (0, 255, 255),   # 青色
        'table_caption': (255, 255, 0),  # 黄色
        'equation': (255, 165, 0),  # 橙色
        'equation_caption': (128, 0, 128)  # 深紫色
    }

    visualized_paths = []

    # 按页面组织块
    pages_blocks = {}
    for block in mineru_output.blocks:
        page_idx = block.page_idx
        if page_idx not in pages_blocks:
            pages_blocks[page_idx] = []
        pages_blocks[page_idx].append(block)

    # 为每个页面生成可视化
    for page_idx, blocks in pages_blocks.items():
        # 获取对应的页面图像
        page_image_info = None
        for img_info in mineru_output.page_images:
            if img_info['page_idx'] == page_idx:
                page_image_info = img_info
                break

        if not page_image_info:
            print(f"警告：找不到第{page_idx + 1}页的图像")
            continue

        # 读取图像
        img_path = Path(page_image_info['image_path'])
        if not img_path.exists():
            print(f"警告：图像文件不存在: {img_path}")
            continue

        # 读取图像（使用cv2.IMREAD_UNCHANGED保持原始通道）
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"错误：无法读取图像 {img_path}")
            continue

        # 如果是RGBA图像，转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 在图像上绘制每个块
        for block in blocks:
            if not block.bbox:
                continue

            # 确定颜色和标签
            color = (128, 128, 128)  # 默认灰色
            label = block.block_type

            if isinstance(block, TextBlock):
                if not draw_text:
                    continue
                if block.text_level > 0:
                    color = colors['title']
                    label = f"Title (Lv{block.text_level})"
                else:
                    color = colors['text']
                    label = "Text"
            elif isinstance(block, ImageBlock):
                if not draw_image:
                    continue
                if block.caption:
                    color = colors['image_caption']
                    label = "Image Caption"
                else:
                    color = colors['image']
                    label = "Image"
            elif isinstance(block, TableBlock):
                if not draw_table:
                    continue
                color = colors['table']
                label = "Table"
            elif isinstance(block, EquationBlock):
                if not draw_equation:
                    continue
                color = colors['equation']
                label = "Equation"

            # 绘制矩形框
            x0, y0, x1, y1 = int(block.bbox.x0), int(block.bbox.y0), int(block.bbox.x1), int(block.bbox.y1)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

            # 绘制标签
            cv2.putText(image, label, (x0, y0 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            content_to_put = block.content[:10] if block.content else ""
            cv2.putText(image, f"Content: {content_to_put}", (x0, y0 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # 保存可视化图像
        output_path = output_dir / f"page_{page_idx + 1:04d}_visualized.png"
        cv2.imwrite(str(output_path), image)
        visualized_paths.append(str(output_path))
        print(f"已保存可视化图像: {output_path}")

        # 打印统计信息
        print(f"\n第{page_idx + 1}页统计:")
        text_count = sum(1 for b in blocks if isinstance(b, TextBlock))
        image_count = sum(1 for b in blocks if isinstance(b, ImageBlock))
        table_count = sum(1 for b in blocks if isinstance(b, TableBlock))
        equation_count = sum(1 for b in blocks if isinstance(b, EquationBlock))
        print(f"  - 文本块: {text_count}个")
        print(f"  - 图像块: {image_count}个")
        print(f"  - 表格块: {table_count}个")
        print(f"  - 公式块: {equation_count}个")

    return visualized_paths
