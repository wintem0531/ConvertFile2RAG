"""文字区域检测和分类工作流测试模块"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 获取项目根目录路径并添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modes.imageTool import image_service  # noqa: E402
from modes.ocrTool import ocr_service  # noqa: E402


def convert_box_to_rect(box: list) -> tuple[int, int, int, int]:
    """
    将OCR返回的box格式转换为矩形框格式

    Args:
        box: OCR返回的box格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        矩形框格式 (x_min, y_min, x_max, y_max)
    """
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))
    return (x_min, y_min, x_max, y_max)


def sort_boxes_left_to_right_top_to_bottom(
    boxes: list[list[list[float]]],
) -> list[list[list[float]]]:
    """
    对检测框进行排序：从左到右、从上到下

    Args:
        boxes: 检测框列表，每个框格式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        排序后的检测框列表
    """
    # 计算每个框的中心点和边界
    box_info = []
    for box in boxes:
        rect_box = convert_box_to_rect(box)
        x_min, y_min, x_max, y_max = rect_box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        box_info.append((box, center_x, center_y, y_min))

    # 排序策略：
    # 1. 首先按 y_min（上边界）排序，允许一定的容差（同一行的框y_min可能略有不同）
    # 2. 在同一行内，按 center_x（中心x坐标）排序

    # 计算平均框高度，用于判断是否在同一行
    if box_info:
        avg_height = np.mean([convert_box_to_rect(box)[3] - convert_box_to_rect(box)[1] for box, _, _, _ in box_info])
        row_tolerance = avg_height * 0.5  # 行容差为平均高度的50%
    else:
        row_tolerance = 50

    # 按 y_min 分组（同一行的框）
    rows = []
    sorted_box_info = sorted(box_info, key=lambda x: (x[3], x[1]))  # 先按y_min，再按x_min

    current_row = []
    current_row_y = None

    for box, center_x, center_y, y_min in sorted_box_info:
        if current_row_y is None or abs(y_min - current_row_y) <= row_tolerance:
            # 同一行
            current_row.append((box, center_x, center_y, y_min))
            if current_row_y is None:
                current_row_y = y_min
        else:
            # 新的一行
            if current_row:
                # 对当前行按 center_x 排序
                current_row.sort(key=lambda x: x[1])
                rows.append([item[0] for item in current_row])
            current_row = [(box, center_x, center_y, y_min)]
            current_row_y = y_min

    # 添加最后一行
    if current_row:
        current_row.sort(key=lambda x: x[1])
        rows.append([item[0] for item in current_row])

    # 展平所有行
    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend(row)

    return sorted_boxes


def test_split_workflow():
    """测试文字区域检测和分类工作流"""
    print("\n" + "=" * 60)
    print("文字区域检测和分类工作流测试")
    print("=" * 60)

    # 测试图像路径
    image_path = PROJECT_ROOT / "test_file/1.pdf2png/all_pages/齊系文字編_page_24.png"
    output_dir = PROJECT_ROOT / "test_file/4.split_workflow"

    if not image_path.exists():
        print(f"❌ 测试文件不存在: {image_path}")
        return

    print(f"📄 测试图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    text_images_dir = output_dir / "text_images"  # 有文字的图像
    non_text_images_dir = output_dir / "non_text_images"  # 无文字的图像
    text_images_dir.mkdir(parents=True, exist_ok=True)
    non_text_images_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 步骤1: 进行文字范围检测（只检测，不识别）
        print("\n🔍 步骤1: 进行文字范围检测...")
        boxes, resized_image, scale = ocr_service.detect_only(image_path)
        print(f"✅ 检测完成，共检测到 {len(boxes)} 个文本框")
        print(f"📏 图像缩放比例: {scale:.4f}")

        if not boxes:
            print("⚠️  未检测到任何文本框")
            return

        # 步骤2: 对所有检测框排序（从左到右、从上到下）
        print("\n📋 步骤2: 对检测框进行排序（从左到右、从上到下）...")
        sorted_boxes = sort_boxes_left_to_right_top_to_bottom(boxes)
        print(f"✅ 排序完成，共 {len(sorted_boxes)} 个检测框")

        # 保存缩放后的图像
        resized_image_path = output_dir / "resized_image.png"
        cv2.imwrite(str(resized_image_path), resized_image)

        # 将检测框转换为矩形格式，供后续使用
        boxes_to_draw = [convert_box_to_rect(box) for box in sorted_boxes]

        # 获取行分组信息（用于后续绘制）
        abnormal_results = image_service.detect_abnormal_boxes(
            image_path=resized_image_path,
            boxes=boxes_to_draw,
            outlier_method="iqr",
            output_path=output_dir / "abnormal_result.png",
        )

        # 创建映射：sorted_boxes的索引 -> 行号和列号信息（用于绘制）
        box_to_result_map: dict[int, dict] = {}
        for sorted_idx, box in enumerate(sorted_boxes):
            rect_box = convert_box_to_rect(box)
            # 在abnormal_results中查找匹配的box
            for result in abnormal_results:
                if result["box"] == rect_box:
                    box_to_result_map[sorted_idx] = result
                    break

        # 用于记录异常框的映射
        box_to_tag_map: dict[int, str] = {}

        # 步骤3: 裁切每个检测框并进行识别
        print("\n✂️  步骤3: 裁切检测框并进行识别...")
        ocr_service_instance = ocr_service.get_ocr_service()

        text_count = 0
        non_text_count = 0

        for idx, box in enumerate(sorted_boxes):
            # 转换box格式为矩形框
            rect_box = convert_box_to_rect(box)
            x_min, y_min, x_max, y_max = rect_box

            # 确保坐标在图像范围内
            h, w = resized_image.shape[:2]
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))

            # 裁切图像区域
            cropped_image = resized_image[y_min:y_max, x_min:x_max]

            if cropped_image.size == 0:
                continue

            # 使用只开启cls和rec的OCR引擎进行识别
            text, confidence, word_boxes = ocr_service_instance.recognize_text_only(cropped_image)
            # word_boxes: 单字坐标列表，格式 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]

            # 获取行号和列号标签
            result_info = box_to_result_map.get(idx, {})
            row_idx = result_info.get("row_index", 0)
            col_idx = result_info.get("col_index", 0)
            label = f"{row_idx}-{col_idx}"

            # 输出日志：标签与OCR结果对应
            text_display = text.strip() if text and text.strip() else "(无文字)"
            print(f"  [{label}] OCR结果: {text_display} (置信度: {confidence:.2f})")

            # 步骤3.5: 结合OCR结果进行异常判定
            is_abnormal = False
            condition_count = 0

            # a. 计算高宽比 aspect_ratio = height / width
            box_height = y_max - y_min
            box_width = x_max - x_min
            aspect_ratio = box_height / box_width if box_width > 0 else 0
            if aspect_ratio > 0.7:
                condition_count += 1

            # b. 统计框内文字识别到的数量，筛选小于2个字的框
            text_length = len(text.strip()) if text else 0
            if text_length < 2:
                condition_count += 1

            # c. 统计框内黑色像素的占比，筛选大于30%的框
            # 转换为灰度图
            if len(cropped_image.shape) == 3:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cropped_image

            # 创建黑色像素的mask（阈值设为128，小于128认为是黑色）
            black_mask = (gray_image < 128).astype(np.uint8) * 255

            # 进行形态学操作：先膨胀再腐蚀，连接相邻的黑色区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # 膨胀操作，连接相邻的黑色像素
            dilated = cv2.dilate(black_mask, kernel, iterations=1)
            # 腐蚀操作，恢复形状
            eroded = cv2.erode(dilated, kernel, iterations=1)

            # 找到黑色区域的bounding box
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 找到所有黑色区域的合并bounding box
                all_points = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_points)

                # 在bounding box范围内计算黑色像素占比
                roi = gray_image[y : y + h, x : x + w]
                roi_total_pixels = roi.size
                roi_black_pixels = np.sum(roi < 128)
                black_ratio = roi_black_pixels / roi_total_pixels if roi_total_pixels > 0 else 0
            else:
                # 如果没有找到黑色区域，占比为0
                black_ratio = 0.0

            if black_ratio > 0.3:
                condition_count += 1

            # 如果满足任意2项及以上，则认为是异常框
            if condition_count >= 2:
                is_abnormal = True
                box_to_tag_map[idx] = "abnormal"
                # 输出异常判定详情
                conditions_met = []
                if aspect_ratio > 0.7:
                    conditions_met.append(f"高宽比={aspect_ratio:.2f}")
                if text_length < 2:
                    conditions_met.append(f"文字数={text_length}")
                if black_ratio > 0.5:
                    conditions_met.append(f"黑色占比={black_ratio:.2%}")
                print(f"    └─ 异常判定: 满足{condition_count}项条件 {', '.join(conditions_met)}")
            else:
                box_to_tag_map[idx] = "normal"

            # 步骤4: 根据识别结果分类存储
            if text and text.strip():  # 有文字
                # 保存到有文字的目录
                image_filename = f"text_{idx:04d}_{text[:10]}_{confidence:.2f}.png"
                # 清理文件名中的非法字符
                image_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in image_filename)
                image_path_save = text_images_dir / image_filename
                cv2.imwrite(str(image_path_save), cropped_image)
                text_count += 1
            else:  # 无文字
                # 保存到无文字的目录
                image_filename = f"non_text_{idx:04d}_{confidence:.2f}.png"
                image_path_save = non_text_images_dir / image_filename
                cv2.imwrite(str(image_path_save), cropped_image)
                non_text_count += 1

            # 如果是异常框，也保存一份到non_text_images目录
            if is_abnormal:
                abnormal_filename = f"abnormal_{idx:04d}.png"
                abnormal_path_save = non_text_images_dir / abnormal_filename
                cv2.imwrite(str(abnormal_path_save), cropped_image)

            # 每处理10个框显示一次进度
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(sorted_boxes)} 个检测框...")

        print(f"✅ 识别完成，共处理 {len(sorted_boxes)} 个检测框")

        # 统计异常框数量
        abnormal_count = sum(1 for tag in box_to_tag_map.values() if tag == "abnormal")
        print(f"✅ 异常检测完成，检测到 {abnormal_count} 个异常框")

        # 统计信息
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("📊 处理结果统计")
        print("=" * 60)
        print(f"📈 检测文本框数量: {len(boxes)}")
        print(f"📈 有文字的图像: {text_count} 个")
        print(f"📈 无文字的图像: {non_text_count} 个")
        print(f"📈 异常框数量: {abnormal_count} 个")
        print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
        print(f"📏 图像缩放比例: {scale:.4f}")
        print(f"📁 有文字图像目录: {text_images_dir}")
        print(f"📁 无文字图像目录: {non_text_images_dir}")
        print(f"🖼️  缩放后的图像: {resized_image_path}")

        # 绘制检测框（可选，用于调试）
        print("\n🎨 绘制检测框（用于调试）...")
        output_image_path = output_dir / "detection_result.png"

        draw_image = cv2.imread(str(resized_image_path))
        if draw_image is None:
            print("⚠️  无法加载缩放后的图像进行标注")
        else:
            # 创建从box到索引的映射，用于查找tag
            box_to_index_map: dict[tuple[int, int, int, int], int] = {}
            for idx, box in enumerate(sorted_boxes):
                rect_box = convert_box_to_rect(box)
                box_to_index_map[rect_box] = idx

            # 使用abnormal_results中的行号和列号信息，但使用box_to_tag_map判断是否为异常框
            for result in abnormal_results:
                x1, y1, x2, y2 = result["box"]
                row_idx = result["row_index"]
                col_idx = result["col_index"]

                # 查找对应的索引，获取实际的tag
                box_key = (x1, y1, x2, y2)
                idx = box_to_index_map.get(box_key, -1)
                tag = box_to_tag_map.get(idx, "normal")

                # 根据tag选择颜色：normal=绿色，abnormal=红色
                color = (0, 0, 255) if tag == "abnormal" else (0, 255, 0)
                thickness = 2

                # 绘制矩形框
                cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, thickness)

                # 绘制标签文本（行号-列号）
                label_text = f"{row_idx}-{col_idx}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

                # 在框上方绘制文本背景
                text_x = x1
                text_y = max(y1 - 5, text_height + 5)
                cv2.rectangle(
                    draw_image,
                    (text_x, text_y - text_height - 5),
                    (text_x + text_width, text_y + baseline),
                    color,
                    -1,
                )

                # 绘制文本
                cv2.putText(
                    draw_image,
                    label_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),  # 白色文本
                    text_thickness,
                    cv2.LINE_AA,
                )

            cv2.imwrite(str(output_image_path), draw_image)
            print(f"✅ 检测结果图像已保存至: {output_image_path}")

        print("\n✅ 工作流测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    """运行测试"""
    print("\n" + "🚀 " * 20)
    print("文字区域检测和分类工作流测试")
    print("🚀 " * 20)

    test_split_workflow()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {PROJECT_ROOT / 'test_file/4.split_workflow'}/")
    print("  - text_images/      : 有文字的图像")
    print("  - non_text_images/  : 无文字的图像")
    print("  - detection_result.png: 检测结果图像（带框线标注）")


if __name__ == "__main__":
    main()
