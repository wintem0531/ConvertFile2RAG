"""PDF 文档分页处理工作流

该工作流处理 PDF 文档的每一页，提取大图像和文本块，
并按照指定的目录结构组织输出结果。
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modes.mineru import mineru_util  # noqa: E402
from modes.imageTool import image_service  # noqa: E402
from modes.ocrTool import ocr_service  # noqa: E402


def convert_box_to_rect(box: list) -> tuple[int, int, int, int]:
    """将OCR返回的box格式转换为矩形框格式"""
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))
    return (x_min, y_min, x_max, y_max)


def sort_boxes_left_to_right_top_to_bottom(boxes: list[list[list[float]]]) -> list[list[list[float]]]:
    """对检测框进行排序：从左到右、从上到下"""
    box_info = []
    for box in boxes:
        rect_box = convert_box_to_rect(box)
        x_min, y_min, x_max, y_max = rect_box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        box_info.append((box, center_x, center_y, y_min))

    # 计算平均框高度
    if box_info:
        avg_height = np.mean([convert_box_to_rect(box)[3] - convert_box_to_rect(box)[1] for box, _, _, _ in box_info])
        row_tolerance = avg_height * 0.5
    else:
        row_tolerance = 50

    # 按行分组
    rows = []
    sorted_box_info = sorted(box_info, key=lambda x: (x[3], x[1]))
    current_row = []
    current_row_y = None

    for box, center_x, center_y, y_min in sorted_box_info:
        if current_row_y is None or abs(y_min - current_row_y) <= row_tolerance:
            current_row.append((box, center_x, center_y, y_min))
            if current_row_y is None:
                current_row_y = y_min
        else:
            if current_row:
                current_row.sort(key=lambda x: x[1])
                rows.append([item[0] for item in current_row])
            current_row = [(box, center_x, center_y, y_min)]
            current_row_y = y_min

    if current_row:
        current_row.sort(key=lambda x: x[1])
        rows.append([item[0] for item in current_row])

    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend(row)

    return sorted_boxes


def process_text_box_with_ocr(
    image: np.ndarray, ocr_instance, output_dir: Path, box_idx: int, image_hashes: dict[str, str]
) -> tuple[str, list[str]]:
    """处理文本框图像，进行OCR识别"""
    text_parts = []
    image_placeholders = []

    # 保存图像到临时文件以便进行OCR检测
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, image)
        tmp_path = Path(tmp_file.name)

    try:
        # 进行文字检测
        boxes, _, _ = ocr_instance.detect_only(tmp_path)

        if not boxes:
            # 没有检测到文字，保存整个图像
            image_bytes = cv2.imencode(".png", image)[1].tobytes()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            image_path = output_dir / f"{image_hash}.png"
            cv2.imwrite(str(image_path), image)
            image_hashes[image_hash] = str(image_path)
            image_placeholders.append(f"{{{image_hash}.png}}")
            return "", image_placeholders

        # 对检测框排序
        sorted_boxes = sort_boxes_left_to_right_top_to_bottom(boxes)

        for idx, box in enumerate(sorted_boxes):
            # 裁切文本区域
            rect_box = convert_box_to_rect(box)
            x_min, y_min, x_max, y_max = rect_box

            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))

            cropped_image = image[y_min:y_max, x_min:x_max]

            if cropped_image.size == 0:
                continue

            # 进行OCR识别
            text, confidence, _ = ocr_instance.recognize_text_only(cropped_image)

            # 判断是否为图像（低置信度或无文字）
            if not text or not text.strip() or confidence < 0.5:
                # 保存为图像
                image_bytes = cv2.imencode(".png", cropped_image)[1].tobytes()
                image_hash = hashlib.md5(image_bytes).hexdigest()
                image_path = output_dir / f"{image_hash}.png"
                cv2.imwrite(str(image_path), cropped_image)
                image_hashes[image_hash] = str(image_path)
                image_placeholders.append(f"{{{image_hash}.png}}")
            else:
                # 保存文本
                text_parts.append(text.strip())

    finally:
        # 清理临时文件
        if tmp_path.exists():
            tmp_path.unlink()

    return " ".join(text_parts), image_placeholders


def process_pdf_page_workflow(
    pdf_input: str | Path | bytes,
    output_dir: str | Path,
    start_page: int = 1,
    end_page: int | None = None,
    min_pixels: int = 10000,
    lang: str = "ch",
    backend: str = "vlm-mlx-engine",
    **kwargs,
) -> dict[str, any]:
    """处理 PDF 文档的工作流，按页面组织输出

    Args:
        pdf_input: PDF 输入（文件路径或字节数据）
        output_dir: 输出目录
        start_page: 开始页码
        end_page: 结束页码
        min_pixels: 大图像最小像素数
        lang: 文档语言
        backend: MinerU 后端引擎

    Returns:
        Dict: 处理结果统计
    """
    start_time = time.time()
    output_dir = Path(output_dir)

    # 处理输入
    pdf_path = None
    pdf_bytes = None
    pdf_filename = "document"

    if isinstance(pdf_input, bytes):
        pdf_bytes = pdf_bytes
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            pdf_path = Path(tmp_file.name)
    else:
        pdf_path = Path(pdf_input)
        pdf_filename = pdf_path.stem

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 创建主输出目录
    pdf_output_dir = output_dir / pdf_filename
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 获取总页数
        import fitz

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()

        if end_page is None:
            end_page = total_pages

        print(f"\n{'=' * 60}")
        print("PDF 分页处理工作流")
        print(f"{'=' * 60}")
        print(f"📄 输入文件: {pdf_path}")
        print(f"📁 输出目录: {pdf_output_dir}")
        print(f"📖 页面范围: {start_page} - {end_page}/{total_pages}")
        print(f"🖼️  大图像阈值: {min_pixels} 像素")

        ocr_instance = ocr_service.get_ocr_service()
        processing_stats = {
            "total_pages": end_page - start_page + 1,
            "processed_pages": 0,
            "total_text_blocks": 0,
            "total_large_images": 0,
            "total_small_images": 0,
            "errors": [],
        }

        # 逐页处理
        for page_num in range(start_page, end_page + 1):
            print(f"\n🔄 处理第 {page_num}/{total_pages} 页...")

            # 创建页面输出目录
            page_dir = pdf_output_dir / str(page_num)
            page_dir.mkdir(parents=True, exist_ok=True)
            images_dir = page_dir / "image"
            images_dir.mkdir(parents=True, exist_ok=True)

            # 创建中间处理目录
            middle_dir = output_dir / f"middle_{page_num}"
            middle_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 步骤1: 使用 MinerU 解析页面
                print("  📋 步骤1: 使用 MinerU 解析页面...")
                content_list = mineru_util.parse_pdf_to_content_list(
                    pdf_path=pdf_path,
                    page_range=(page_num, page_num),
                    output_dir=middle_dir,
                    lang=lang,
                    backend=backend,
                    **kwargs,
                )

                # 过滤内容块，排除页码等不需要的内容
                content_blocks = []
                for item in content_list:
                    item_type = item.get("type", "")
                    # 只保留文本、标题和图像
                    if item_type in ["text", "title", "header", "image"]:
                        content_blocks.append(item)

                image_count = len([b for b in content_blocks if b.get("type") == "image"])
                print(f"    ✓ 提取到 {len(content_blocks)} 个内容块（{image_count} 个图像）")

                # 收集所有文本和图像
                result_parts = []
                image_hashes = {}  # 存储图像哈希到路径的映射
                text_box_idx = 1

                # 步骤2: 处理每个内容块
                print("  📝 步骤2: 处理内容块...")
                for block_idx, block in enumerate(content_blocks):
                    block_type = block.get("type", "")

                    if block_type == "image":
                        # 处理图像块
                        img_path = block.get("img_path", "")
                        if img_path:
                            # content_list 的图像路径在 vlm 目录下
                            full_img_path = middle_dir / pdf_filename / "vlm" / img_path
                            if not full_img_path.exists():
                                # 尝试其他可能的路径
                                full_img_path = middle_dir / "vlm" / img_path

                            if full_img_path.exists():
                                # 读取图像并检查大小
                                image = cv2.imread(str(full_img_path))
                                if image is not None:
                                    h, w = image.shape[:2]
                                    total_pixels = h * w

                                    if total_pixels >= min_pixels:
                                        # 大图像：生成哈希并保存
                                        print(f"    🖼️  发现大图像: {w}x{h} ({total_pixels:,} 像素)")
                                        image_bytes = cv2.imencode(".png", image)[1].tobytes()
                                        image_hash = hashlib.md5(image_bytes).hexdigest()
                                        output_img_path = images_dir / f"{image_hash}.png"
                                        cv2.imwrite(str(output_img_path), image)
                                        image_hashes[image_hash] = str(output_img_path)
                                        result_parts.append(f"{{{image_hash}.png}}")
                                        processing_stats["total_large_images"] += 1

                                        # 打印图像说明（如果有）
                                        captions = block.get("image_caption", [])
                                        if captions:
                                            print(f"      说明: {', '.join(captions)}")
                                    else:
                                        # 小图像：保存为文本框图像
                                        print(f"    📄 发现小图像: {w}x{h} ({total_pixels:,} 像素)")
                                        text_box_path = page_dir / f"text_box_{text_box_idx}.png"
                                        cv2.imwrite(str(text_box_path), image)
                                        text_box_idx += 1
                                        processing_stats["total_small_images"] += 1

                    elif block_type in ["text", "title", "header"]:
                        # 处理文本块
                        text_content = block.get("text", "")
                        if text_content:
                            result_parts.append(text_content)
                            processing_stats["total_text_blocks"] += 1
                            print(f"    📄 文本块: {text_content[:50]}...")

                # 步骤3: 处理文本框图像
                print("  🔍 步骤3: 处理文本框图像...")
                text_box_files = sorted(page_dir.glob("text_box_*.png"))

                for text_box_path in text_box_files:
                    print(f"    📷 处理: {text_box_path.name}")
                    image = cv2.imread(str(text_box_path))

                    # 使用OCR处理文本框
                    text, image_placeholders = process_text_box_with_ocr(
                        image=image,
                        ocr_instance=ocr_instance,
                        output_dir=images_dir,
                        box_idx=int(text_box_path.stem.split("_")[-1]),
                        image_hashes=image_hashes,
                    )

                    # 将结果插入到适当位置
                    # 这里简单追加，实际可能需要根据位置信息插入
                    if text:
                        result_parts.append(text)
                    if image_placeholders:
                        result_parts.extend(image_placeholders)

                    # 删除原始文本框文件
                    text_box_path.unlink()

                # 步骤4: 保存文本结果
                print("  💾 步骤4: 保存文本结果...")
                text_output = "\n".join(str(part) for part in result_parts)
                text_file_path = page_dir / "text.txt"
                text_file_path.write_text(text_output, encoding="utf-8")

                # 清理中间目录
                import shutil

                if middle_dir.exists():
                    shutil.rmtree(middle_dir, ignore_errors=True)

                processing_stats["processed_pages"] += 1
                print(f"  ✅ 第 {page_num} 页处理完成")

            except Exception as e:
                error_msg = f"处理第 {page_num} 页时出错: {str(e)}"
                print(f"  ❌ {error_msg}")
                processing_stats["errors"].append({"page": page_num, "error": error_msg})

                # 清理中间目录
                import shutil

                if middle_dir.exists():
                    shutil.rmtree(middle_dir, ignore_errors=True)
                continue

        # 保存整体统计信息
        stats_file = pdf_output_dir / "processing_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processing_info": {
                        "pdf_path": str(pdf_path),
                        "start_page": start_page,
                        "end_page": end_page,
                        "min_pixels": min_pixels,
                        "processing_time": round(time.time() - start_time, 2),
                    },
                    "statistics": processing_stats,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 打印最终统计
        print(f"\n{'=' * 60}")
        print("📊 处理完成统计")
        print(f"{'=' * 60}")
        print(f"📄 处理页面: {processing_stats['processed_pages']}/{processing_stats['total_pages']}")
        print(f"📝 文本块: {processing_stats['total_text_blocks']} 个")
        print(f"🖼️  大图像: {processing_stats['total_large_images']} 个")
        print(f"📄 小图像(作为文本框): {processing_stats['total_small_images']} 个")
        print(f"❌ 错误数: {len(processing_stats['errors'])}")
        print(f"⏱️  总耗时: {round(time.time() - start_time, 2)} 秒")
        print("\n📁 输出结构:")
        print(f"  {pdf_output_dir}/")
        print("  ├── 1/")
        print("  │   ├── text.txt")
        print("  │   └── image/")
        print("  │       └── {hash}.png")
        print("  ├── 2/")
        print("  │   └── ...")
        print("  └── processing_stats.json")

        return processing_stats

    finally:
        # 清理临时文件
        if isinstance(pdf_input, bytes) and pdf_path and pdf_path.exists():
            import os

            try:
                os.unlink(pdf_path)
            except Exception:
                pass


def main():
    """主函数 - 演示工作流"""
    print("\n🚀 PDF 分页处理工作流演示")

    # 查找测试文件
    test_dir = PROJECT_ROOT / "test_file"
    test_files = list(test_dir.rglob("*.pdf"))[:1]

    if not test_files:
        test_files = list(test_dir.rglob("*.png"))[:1]

    if not test_files:
        print("❌ 未找到测试文件")
        return

    test_file = test_files[0]
    output_dir = PROJECT_ROOT / "test_file" / "split_workflow_output"

    try:
        process_pdf_page_workflow(
            pdf_input=test_file,
            output_dir=output_dir,
            start_page=25,
            end_page=25,  # 只处理第25页
            min_pixels=10000,  # 使用默认阈值
            lang="ch",
            backend="vlm-mlx-engine",
        )

        print("\n✅ 工作流执行完成!")

    except Exception as e:
        print(f"\n❌ 工作流执行失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
