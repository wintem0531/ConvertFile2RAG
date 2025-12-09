"""PDF转图像服务模块"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF


def get_optimal_workers() -> int:
    """
    自动检测系统配置并返回最优的并发工作线程数

    Returns:
        最优的并发工作线程数（通常是CPU核心数+1，最小为1）
    """
    cpu_count = os.cpu_count() or 1
    # 对于I/O密集型任务，通常使用 CPU核心数 + 1
    # 对于CPU密集型任务，使用 CPU核心数
    # PDF转图像是混合型任务，使用 CPU核心数 + 1 作为默认值
    optimal_workers = min(cpu_count + 1, 16)  # 限制最大并发数为16，避免过度并发
    return max(optimal_workers, 1)  # 至少1个worker


def _convert_page_to_image(
    pdf_path: Path,
    page_num: int,
    output_dir: Path,
    pdf_name: str,
    dpi: int,
    image_format: str,
) -> str:
    """
    内部函数：将单个PDF页面转换为图像（用于并行处理）

    Args:
        pdf_path: PDF文件路径
        page_num: 页码（0索引）
        output_dir: 输出目录路径
        pdf_name: PDF文件名（不含扩展名）
        dpi: 图像分辨率
        image_format: 图像格式

    Returns:
        生成的图像文件路径
    """
    pdf_document = fitz.open(pdf_path)
    try:
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

        image_filename = f"{pdf_name}_page_{page_num + 1}.{image_format}"
        image_path = output_dir / image_filename
        pix.save(str(image_path))

        return str(image_path)
    finally:
        pdf_document.close()


def pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    image_format: str = "png",
    max_workers: int | None = None,
    use_async: bool = True,
) -> list[str]:
    """
    将PDF的所有页面转换为图像（支持异步并行处理）

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录路径
        dpi: 图像分辨率，默认200
        image_format: 图像格式，默认png（支持png、jpg等）
        max_workers: 最大并发工作线程数，None时自动检测
        use_async: 是否使用异步并行处理，默认True

    Returns:
        生成的图像文件路径列表（按页码顺序）

    Raises:
        FileNotFoundError: PDF文件不存在
        ValueError: 输出目录无效或无法创建
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 先打开PDF获取总页数
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    pdf_name = pdf_path.stem
    pdf_document.close()

    if total_pages == 0:
        return []

    # 如果只有1页或禁用异步，使用同步处理
    if total_pages == 1 or not use_async:
        return _pdf_to_images_sync(pdf_path, output_dir, dpi, image_format, pdf_name)

    # 使用异步并行处理
    workers = max_workers if max_workers is not None else get_optimal_workers()
    return asyncio.run(_pdf_to_images_async(pdf_path, output_dir, dpi, image_format, pdf_name, total_pages, workers))


def _pdf_to_images_sync(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    image_format: str,
    pdf_name: str,
) -> list[str]:
    """同步版本：顺序处理所有页面"""
    pdf_document = fitz.open(pdf_path)
    image_paths = []

    try:
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

            image_filename = f"{pdf_name}_page_{page_num + 1}.{image_format}"
            image_path = output_dir / image_filename
            pix.save(str(image_path))
            image_paths.append(str(image_path))
    finally:
        pdf_document.close()

    return image_paths


async def _pdf_to_images_async(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    image_format: str,
    pdf_name: str,
    total_pages: int,
    max_workers: int,
) -> list[str]:
    """异步版本：并行处理所有页面"""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建所有页面的转换任务
        tasks = [
            loop.run_in_executor(
                executor,
                _convert_page_to_image,
                pdf_path,
                page_num,
                output_dir,
                pdf_name,
                dpi,
                image_format,
            )
            for page_num in range(total_pages)
        ]

        # 并行执行所有任务，保持顺序
        image_paths = await asyncio.gather(*tasks)

    return list(image_paths)


def pdf_page_to_image(
    pdf_path: str | Path,
    page_number: int,
    output_dir: str | Path,
    dpi: int = 200,
    image_format: str = "png",
) -> str:
    """
    将PDF的指定页转换为图像

    Args:
        pdf_path: PDF文件路径
        page_number: 页码（从1开始）
        output_dir: 输出目录路径
        dpi: 图像分辨率，默认200
        image_format: 图像格式，默认png

    Returns:
        生成的图像文件路径

    Raises:
        FileNotFoundError: PDF文件不存在
        ValueError: 页码无效
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_document = fitz.open(pdf_path)

    try:
        if page_number < 1 or page_number > len(pdf_document):
            raise ValueError(f"页码无效: {page_number}，PDF共有 {len(pdf_document)} 页")

        page = pdf_document[page_number - 1]  # 转换为0索引
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

        pdf_name = pdf_path.stem
        image_filename = f"{pdf_name}_page_{page_number}.{image_format}"
        image_path = output_dir / image_filename
        pix.save(str(image_path))

        return str(image_path)
    finally:
        pdf_document.close()


def pdf_pages_range_to_images(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
    output_dir: str | Path,
    dpi: int = 200,
    image_format: str = "png",
    max_workers: int | None = None,
    use_async: bool = True,
) -> list[str]:
    """
    将PDF的指定页数范围转换为图像（支持异步并行处理）

    Args:
        pdf_path: PDF文件路径
        start_page: 起始页码（从1开始，包含）
        end_page: 结束页码（从1开始，包含）
        output_dir: 输出目录路径
        dpi: 图像分辨率，默认200
        image_format: 图像格式，默认png
        max_workers: 最大并发工作线程数，None时自动检测
        use_async: 是否使用异步并行处理，默认True

    Returns:
        生成的图像文件路径列表（按页码顺序）

    Raises:
        FileNotFoundError: PDF文件不存在
        ValueError: 页码范围无效
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 先打开PDF获取总页数并验证范围
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    pdf_name = pdf_path.stem

    try:
        if start_page < 1 or end_page < 1:
            raise ValueError("页码必须从1开始")

        if start_page > end_page:
            raise ValueError(f"起始页码 ({start_page}) 不能大于结束页码 ({end_page})")

        if end_page > total_pages:
            raise ValueError(f"结束页码 ({end_page}) 超出PDF总页数 ({total_pages})")
    finally:
        pdf_document.close()

    page_count = end_page - start_page + 1

    # 如果只有1页或禁用异步，使用同步处理
    if page_count == 1 or not use_async:
        return _pdf_pages_range_to_images_sync(pdf_path, output_dir, dpi, image_format, pdf_name, start_page, end_page)

    # 使用异步并行处理
    workers = max_workers if max_workers is not None else get_optimal_workers()
    return asyncio.run(
        _pdf_pages_range_to_images_async(
            pdf_path,
            output_dir,
            dpi,
            image_format,
            pdf_name,
            start_page,
            end_page,
            workers,
        )
    )


def _pdf_pages_range_to_images_sync(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    image_format: str,
    pdf_name: str,
    start_page: int,
    end_page: int,
) -> list[str]:
    """同步版本：顺序处理指定范围的页面"""
    pdf_document = fitz.open(pdf_path)
    image_paths = []

    try:
        for page_num in range(start_page - 1, end_page):  # 转换为0索引
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

            image_filename = f"{pdf_name}_page_{page_num + 1}.{image_format}"
            image_path = output_dir / image_filename
            pix.save(str(image_path))
            image_paths.append(str(image_path))
    finally:
        pdf_document.close()

    return image_paths


async def _pdf_pages_range_to_images_async(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    image_format: str,
    pdf_name: str,
    start_page: int,
    end_page: int,
    max_workers: int,
) -> list[str]:
    """异步版本：并行处理指定范围的页面"""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建指定范围页面的转换任务
        tasks = [
            loop.run_in_executor(
                executor,
                _convert_page_to_image,
                pdf_path,
                page_num - 1,  # 转换为0索引
                output_dir,
                pdf_name,
                dpi,
                image_format,
            )
            for page_num in range(start_page, end_page + 1)
        ]

        # 并行执行所有任务，保持顺序
        image_paths = await asyncio.gather(*tasks)

    return list(image_paths)
