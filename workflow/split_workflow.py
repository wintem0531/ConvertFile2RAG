"""
MinerU 图片处理工作流

该模块提供了使用 MinerU 处理图片并支持页面范围的功能。
"""

import os
import sys
from pathlib import Path


def mineru_get_large_pic(
    image_path,
    output_dir,
    start_page_id=0,
    end_page_id=None,
    backend="vlm-mlx-engine",
    p_lang_list=None,
    p_formula_enable=True,
    p_table_enable=True,
    f_draw_layout_bbox=True,
    f_draw_span_bbox=True,
    f_dump_md=True,
    f_dump_middle_json=True,
    f_dump_model_output=True,
    f_dump_orig_pdf=True,
    f_dump_content_list=True,
):
    """
    使用 MinerU 处理图片，支持指定页面范围

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        start_page_id: 起始页面ID（默认为0）
        end_page_id: 结束页面ID（默认为None，表示处理到最后一页）
        backend: 处理后端（默认为vlm-mlx-engine）
        p_lang_list: 语言列表（默认为["ch"]）
        p_formula_enable: 是否启用公式识别
        p_table_enable: 是否启用表格识别
        f_draw_layout_bbox: 是否绘制布局边界框
        f_draw_span_bbox: 是否绘制跨度边界框
        f_dump_md: 是否输出Markdown
        f_dump_middle_json: 是否输出中间JSON
        f_dump_model_output: 是否输出模型原始输出
        f_dump_orig_pdf: 是否输出原始PDF
        f_dump_content_list: 是否输出内容列表

    Returns:
        Path: 处理结果保存的路径
    """
    # 设置项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from mineru.cli.common import do_parse, read_fn

    # 设置默认语言为中文
    if p_lang_list is None:
        p_lang_list = ["ch"]

    # 设置环境变量以尝试使用MPS（如果可用）
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 将图片转换为PDF字节（MinerU需要PDF格式）
    pdf_bytes = read_fn(image_path)
    pdf_file_name = Path(image_path).stem

    # 调用do_parse进行分析，添加页面范围参数
    do_parse(
        output_dir=str(output_dir),
        pdf_file_names=[pdf_file_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=p_lang_list,
        backend=backend,
        parse_method="vlm",
        p_formula_enable=p_formula_enable,
        p_table_enable=p_table_enable,
        f_draw_layout_bbox=f_draw_layout_bbox,
        f_draw_span_bbox=f_draw_span_bbox,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_pdf=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )

    # 返回处理结果的路径
    result_path = Path(output_dir) / pdf_file_name / "vlm"
    return result_path


def main():
    # 示例用法
    image_path = "test_file/1.pdf2png/all_pages/齊系文字編_page_25.png"
    output_dir = "test_file/5.mineru"

    # 调用函数，处理第0页到第0页（即只处理第一页）
    result_path = mineru_get_large_pic(
        image_path=image_path,
        output_dir=output_dir,
        start_page_id=0,
        end_page_id=0,  # 只处理第一页
    )

    print(f"处理结果保存在: {result_path}")


if __name__ == "__main__":
    main()
