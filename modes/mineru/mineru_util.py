"""MinerU 工具函数封装模块

该模块封装了 MinerU 库的调用，提供了便捷的接口用于 PDF 文档解析和内容提取。
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mineru.cli.common import do_parse, read_fn
    from mineru.utils.enum_class import MakeMode
except ImportError:
    raise ImportError("无法导入 MinerU 库，请确保已正确安装 MinerU")


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
    content_list = parse_pdf_to_content_list(
        pdf_path=pdf_path, page_range=page_range, output_dir=output_dir, lang=lang, backend=backend, **kwargs
    )

    return extract_text_from_content_list(content_list)


# 便捷函数：直接从 PDF 文件提取内容列表
def extract_content_list_from_pdf(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    save_result: bool = False,
    save_path: str | Path | None = None,
    **kwargs,
) -> list[dict]:
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
        **kwargs: 其他传递给 do_parse 的参数

    Returns:
        List[Dict]: 内容列表
    """
    content_list = parse_pdf_to_content_list(
        pdf_path=pdf_path, page_range=page_range, output_dir=output_dir, lang=lang, backend=backend, **kwargs
    )

    if save_result:
        if save_path is None:
            pdf_path = Path(pdf_path)
            save_path = pdf_path.parent / f"{pdf_path.stem}_content_list.json"

        save_content_list_to_json(content_list, save_path)

    return content_list
