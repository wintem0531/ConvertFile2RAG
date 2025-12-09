"""图像处理服务模块"""

from pathlib import Path

import cv2
import numpy as np


def color_to_grayscale(
    image_path: str | Path,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    将彩色图像转换为灰度图

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径，如果为None则只返回变量，不保存

    Returns:
        灰度图像的numpy数组 (H, W)

    Raises:
        FileNotFoundError: 图像文件不存在
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 转换为灰度图
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), grayscale)

    return grayscale


def grayscale_to_binary(
    image_path: str | Path,
    threshold: int = 128,
    output_path: str | Path | None = None,
    use_otsu: bool = False,
) -> np.ndarray:
    """
    将灰度图转换为二值图

    Args:
        image_path: 输入灰度图像路径
        threshold: 二值化阈值（0-255），默认128
        output_path: 输出图像路径，如果为None则只返回变量，不保存
        use_otsu: 是否使用Otsu自适应阈值，默认False

    Returns:
        二值图像的numpy数组 (H, W)，值为0或255

    Raises:
        FileNotFoundError: 图像文件不存在
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取灰度图像
    grayscale = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 二值化
    if use_otsu:
        _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), binary)

    return binary


def resize_image_by_max_side(
    image_path: str | Path,
    max_side: int = 2000,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, float]:
    """
    按最长边等比例缩放图像

    Args:
        image_path: 输入图像路径
        max_side: 最长边的最大像素值，默认2000
        output_path: 输出图像路径，如果为None则只返回变量，不保存

    Returns:
        (缩放后的图像numpy数组, 缩放比例)

    Raises:
        FileNotFoundError: 图像文件不存在
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    h, w = image.shape[:2]
    max_dimension = max(h, w)

    # 如果最长边已经小于等于max_side，不需要缩放
    if max_dimension <= max_side:
        scale = 1.0
        resized = image
    else:
        # 计算缩放比例
        scale = max_side / max_dimension
        new_width = int(w * scale)
        new_height = int(h * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), resized)

    return resized, scale


def resize_images(
    image_paths: list[str | Path],
    target_size: int,
    dimension: str = "width",
    output_dir: str | Path | None = None,
) -> list[np.ndarray]:
    """
    按长或宽的长度，按比例缩放一批图像

    Args:
        image_paths: 输入图像路径列表
        target_size: 目标尺寸（像素）
        dimension: 缩放维度，'width' 或 'height'，默认'width'
        output_dir: 输出目录，如果为None则只返回变量，不保存

    Returns:
        缩放后的图像numpy数组列表

    Raises:
        ValueError: dimension参数无效
    """
    if dimension not in ("width", "height"):
        raise ValueError("dimension必须是'width'或'height'")

    resized_images = []
    output_dir = Path(output_dir) if output_dir else None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"警告: 图像文件不存在，跳过: {image_path}")
            continue

        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告: 无法读取图像文件，跳过: {image_path}")
            continue

        h, w = image.shape[:2]

        # 计算缩放比例
        if dimension == "width":
            scale = target_size / w
            new_width = target_size
            new_height = int(h * scale)
        else:  # dimension == "height"
            scale = target_size / h
            new_height = target_size
            new_width = int(w * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized)

        # 如果指定了输出目录，保存图像
        if output_dir:
            output_filename = f"{image_path.stem}_resized_{dimension}{target_size}{image_path.suffix}"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), resized)

    return resized_images


def morphology_operation(
    image_path: str | Path,
    operation: str = "dilate",
    kernel_size: int = 3,
    iterations: int = 1,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    形态学操作（膨胀和腐蚀）

    Args:
        image_path: 输入图像路径
        operation: 操作类型，'dilate'（膨胀）或'erode'（腐蚀），默认'dilate'
        kernel_size: 核大小，必须是奇数，默认3
        iterations: 操作轮次，默认1
        output_path: 输出图像路径，如果为None则只返回变量，不保存

    Returns:
        处理后的图像numpy数组

    Raises:
        FileNotFoundError: 图像文件不存在
        ValueError: operation或kernel_size参数无效
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    if operation not in ("dilate", "erode"):
        raise ValueError("operation必须是'dilate'或'erode'")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size必须是奇数")

    # 读取图像
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 创建形态学核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 执行形态学操作
    if operation == "dilate":
        result = cv2.dilate(image, kernel, iterations=iterations)
    else:  # operation == "erode"
        result = cv2.erode(image, kernel, iterations=iterations)

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

    return result


def crop_image(
    image_path: str | Path,
    box: tuple[int, int, int, int] | None = None,
    corners: tuple[int, int, int, int] | list[tuple[int, int]] | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    按照box或者对角角点裁切图像

    Args:
        image_path: 输入图像路径
        box: 裁切框，格式为 (x1, y1, x2, y2) 或 (left, top, right, bottom)
        corners: 对角角点，可以是：
            - (x1, y1, x2, y2) 元组
            - [(x1, y1), (x2, y2)] 列表，包含两个角点
        output_path: 输出图像路径，如果为None则只返回变量，不保存

    Returns:
        裁切后的图像numpy数组

    Raises:
        FileNotFoundError: 图像文件不存在
        ValueError: box和corners都未提供，或参数无效
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    if box is None and corners is None:
        raise ValueError("必须提供box或corners参数之一")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    h, w = image.shape[:2]

    # 确定裁切坐标
    if box is not None:
        x1, y1, x2, y2 = box
    else:  # corners is not None
        if isinstance(corners, tuple) and len(corners) == 4:
            # 格式: (x1, y1, x2, y2)
            x1, y1, x2, y2 = corners
        elif isinstance(corners, list) and len(corners) == 2:
            # 格式: [(x1, y1), (x2, y2)]
            (x1, y1), (x2, y2) = corners
        else:
            raise ValueError("corners格式无效，应为(x1, y1, x2, y2)或[(x1, y1), (x2, y2)]")

    # 确保坐标在图像范围内
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # 确保 x1 < x2, y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # 裁切图像
    cropped = image[y1:y2, x1:x2]

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)

    return cropped


def draw_boxes(
    image_path: str | Path,
    boxes: list[tuple[int, int, int, int]] | None = None,
    box: tuple[int, int, int, int] | None = None,
    corners: (
        tuple[int, int, int, int]
        | list[tuple[int, int]]
        | list[tuple[int, int, int, int] | list[tuple[int, int]]]
        | None
    ) = None,
    output_path: str | Path | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str | None = None,
    labels: list[str] | None = None,
) -> np.ndarray:
    """
    根据box或者对角角点，在图中绘制出对应的框线

    Args:
        image_path: 输入图像路径
        boxes: box列表，每个box格式为 (x1, y1, x2, y2)，用于绘制多个框
        box: 单个box，格式为 (x1, y1, x2, y2)，用于绘制单个框
        corners: 对角角点，可以是：
            - (x1, y1, x2, y2) 元组 - 单个框
            - [(x1, y1), (x2, y2)] 列表 - 单个框
            - 包含多个框的列表 - 多个框
        output_path: 输出图像路径，如果为None则只返回变量，不保存
        color: 框线颜色 (B, G, R)，默认绿色 (0, 255, 0)
        thickness: 框线粗细，默认2
        label: 单个框的标签文本（显示在框上方）
        labels: 多个框的标签文本列表（与boxes对应）

    Returns:
        绘制了框线的图像numpy数组

    Raises:
        FileNotFoundError: 图像文件不存在
        ValueError: 参数无效
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    h, w = image.shape[:2]

    # 收集所有要绘制的框
    boxes_to_draw = []

    # 处理boxes参数（多个框）
    if boxes is not None:
        if not isinstance(boxes, list):
            raise ValueError("boxes必须是列表")
        boxes_to_draw.extend(boxes)
        if labels is not None and len(labels) != len(boxes):
            raise ValueError("labels长度必须与boxes相同")

    # 处理box参数（单个框）
    if box is not None:
        boxes_to_draw.append(box)
        if label is not None:
            labels = [label] if labels is None else labels + [label]

    # 处理corners参数
    if corners is not None:
        if isinstance(corners, tuple) and len(corners) == 4:
            # 格式: (x1, y1, x2, y2) - 单个框
            x1, y1, x2, y2 = corners
            boxes_to_draw.append((x1, y1, x2, y2))
            if label is not None:
                labels = [label] if labels is None else labels + [label]
        elif isinstance(corners, list):
            if len(corners) == 2 and all(isinstance(c, (tuple, list)) and len(c) == 2 for c in corners):
                # 格式: [(x1, y1), (x2, y2)] - 单个框
                (x1, y1), (x2, y2) = corners
                boxes_to_draw.append((x1, y1, x2, y2))
                if label is not None:
                    labels = [label] if labels is None else labels + [label]
            else:
                # 多个框的列表
                for corner_item in corners:
                    if isinstance(corner_item, tuple) and len(corner_item) == 4:
                        # 格式: (x1, y1, x2, y2)
                        boxes_to_draw.append(corner_item)
                    elif isinstance(corner_item, list) and len(corner_item) == 2:
                        # 格式: [(x1, y1), (x2, y2)]
                        (x1, y1), (x2, y2) = corner_item
                        boxes_to_draw.append((x1, y1, x2, y2))
                    else:
                        raise ValueError(f"corners格式无效: {corner_item}，应为(x1, y1, x2, y2)或[(x1, y1), (x2, y2)]")

    if not boxes_to_draw:
        raise ValueError("必须提供boxes、box或corners参数之一")

    # 绘制所有框
    for idx, box_coords in enumerate(boxes_to_draw):
        if not isinstance(box_coords, tuple) or len(box_coords) != 4:
            raise ValueError(f"box格式无效: {box_coords}，应为(x1, y1, x2, y2)")

        x1, y1, x2, y2 = box_coords

        # 确保坐标在图像范围内
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        # 确保 x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 如果有标签，在框上方绘制文本
        if labels is not None and idx < len(labels) and labels[idx]:
            label_text = str(labels[idx])
            # 计算文本大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

            # 在框上方绘制文本背景
            text_x = x1
            text_y = max(y1 - 5, text_height + 5)
            cv2.rectangle(
                image,
                (text_x, text_y - text_height - 5),
                (text_x + text_width, text_y + baseline),
                color,
                -1,
            )

            # 绘制文本
            cv2.putText(
                image,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),  # 白色文本
                text_thickness,
                cv2.LINE_AA,
            )

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    return image


def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    overlap_threshold: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """
    按照box是否重合，合并同一张图像上的box列表

    Args:
        boxes: box列表，每个box格式为 (x1, y1, x2, y2)
        overlap_threshold: 重叠阈值（IoU），默认0.5

    Returns:
        合并后的box列表

    Raises:
        ValueError: boxes为空或格式无效
    """
    if not boxes:
        return []

    # 验证box格式
    for box in boxes:
        if not isinstance(box, tuple) or len(box) != 4:
            raise ValueError(f"box格式无效: {box}，应为(x1, y1, x2, y2)")

    def calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        """计算两个box的IoU（交并比）"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def merge_two_boxes(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """合并两个box"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1 = min(x1_1, x1_2)
        y1 = min(y1_1, y1_2)
        x2 = max(x2_1, x2_2)
        y2 = max(y2_1, y2_2)

        return (x1, y1, x2, y2)

    # 合并重叠的box
    merged = []
    used = [False] * len(boxes)

    for i, box1 in enumerate(boxes):
        if used[i]:
            continue

        current_box = box1
        used[i] = True

        # 查找所有与当前box重叠的box并合并
        changed = True
        while changed:
            changed = False
            for j, box2 in enumerate(boxes):
                if used[j] or i == j:
                    continue

                iou = calculate_iou(current_box, box2)
                if iou >= overlap_threshold:
                    current_box = merge_two_boxes(current_box, box2)
                    used[j] = True
                    changed = True

        merged.append(current_box)

    return merged
