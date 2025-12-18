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

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mineru.cli.common import do_parse, read_fn
    from mineru.utils.enum_class import MakeMode
except ImportError:
    raise ImportError("无法导入 MinerU 库，请确保已正确安装 MinerU")


@dataclass
class BoundingBox:
    """边界框信息"""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class ContentBlock:
    """基础内容块"""
    content: str
    block_type: str  # text, image, table, equation, etc.
    bbox: BoundingBox | None = None
    page_idx: int = 0

    def __str__(self) -> str:
        return f"{self.block_type}: {self.content[:100]}{'...' if len(self.content) > 100 else ''}"


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

    @property
    def image_caption(self) -> list[str]:
        return self.caption

    @property
    def image_footnote(self) -> list[str]:
        return self.footnote


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
        """获取指定页面的所有块"""
        return [block for block in self.blocks if block.page_idx == page_idx]

    def add_block(self, block: ContentBlock):
        """添加内容块"""
        self.blocks.append(block)


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
                    y1=float(bbox_coords[3])
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

            text_block = TextBlock(
                content=item.get("text", ""),
                bbox=bbox,
                page_idx=page_idx,
                text_level=text_level
            )
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
                footnote=item.get("image_footnote", [])
            )
            output.add_block(image_block)

        elif block_type == "table":
            # 创建表格块
            table_block = TableBlock(
                content=item.get("text", ""),
                bbox=bbox,
                page_idx=page_idx,
                markdown=item.get("table_caption", "")
            )
            output.add_block(table_block)

        elif block_type == "equation":
            # 创建公式块
            equation_block = EquationBlock(
                content=item.get("text", ""),
                bbox=bbox,
                page_idx=page_idx,
                latex=item.get("latex", "")
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
                        y1=float(bbox_coords[3])
                    )

            # 获取块类型
            block_type = block_data.get("type", "text")

            # 根据类型创建不同的块
            if block_type in ["text", "title", "subtitle"]:
                text_level = block_data.get("level", 0)
                if block_type in ["title", "subtitle"]:
                    text_level = text_level + 1

                text_block = TextBlock(
                    content=block_data.get("text", ""),
                    bbox=bbox,
                    page_idx=page_idx,
                    text_level=text_level
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
                    footnote=block_data.get("footnotes", [])
                )
                output.add_block(image_block)

            elif block_type == "table":
                table_block = TableBlock(
                    content=block_data.get("text", ""),
                    bbox=bbox,
                    page_idx=page_idx,
                    markdown=block_data.get("markdown", "")
                )
                output.add_block(table_block)

            elif block_type == "equation":
                equation_block = EquationBlock(
                    content=block_data.get("text", ""),
                    bbox=bbox,
                    page_idx=page_idx,
                    latex=block_data.get("latex", "")
                )
                output.add_block(equation_block)

    return output


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
        return _convert_content_list_to_blocks(content_list)

    elif return_format == "middle_json":
        # 使用 middle_json 格式解析
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
        return _convert_middle_json_to_blocks(middle_json)

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
