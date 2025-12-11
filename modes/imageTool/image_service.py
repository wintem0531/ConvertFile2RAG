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
    containment_ratio: float = 0.8,
) -> list[tuple[int, int, int, int]]:
    """
    按照box是否重合，合并同一张图像上的box列表
    支持IoU重叠和包含关系两种合并策略

    Args:
        boxes: box列表，每个box格式为 (x1, y1, x2, y2)
        overlap_threshold: 重叠阈值（IoU），默认0.5
        containment_ratio: 包含度阈值，当小框被大框包含的比例超过此值时合并，默认0.8

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

    def is_contained(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
        """
        检查box1是否完全包含box2（box2在box1内部）

        Returns:
            True if box2 is completely inside box1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        return x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2

    def calculate_containment_ratio(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        """
        计算小框被大框包含的比例（intersection / min(area1, area2)）
        用于判断小框是否大部分在大框内

        Returns:
            包含度比例（0-1），值越大表示小框越被大框包含
        """
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

        # 计算两个box的面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        min_area = min(area1, area2)

        if min_area == 0:
            return 0.0

        # 返回交集占较小框的比例
        return intersection / min_area

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

                # 策略1：检查完全包含关系
                if is_contained(current_box, box2) or is_contained(box2, current_box):
                    current_box = merge_two_boxes(current_box, box2)
                    used[j] = True
                    changed = True
                    continue

                # 策略2：检查IoU重叠
                iou = calculate_iou(current_box, box2)
                if iou >= overlap_threshold:
                    current_box = merge_two_boxes(current_box, box2)
                    used[j] = True
                    changed = True
                    continue

                # 策略3：检查包含度（小框大部分在大框内）
                containment = calculate_containment_ratio(current_box, box2)
                if containment >= containment_ratio:
                    current_box = merge_two_boxes(current_box, box2)
                    used[j] = True
                    changed = True

        merged.append(current_box)

    return merged


def detect_text_regions_morphology(
    image_path: str | Path,
    min_area: int = 100,
    max_area: int = 10000,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
    output_path: str | Path | None = None,
) -> list[dict[str, tuple[int, int, int, int] | tuple[float, float]]]:
    """
    通过形态学操作检测文字区域，获得box及中心坐标

    Args:
        image_path: 输入图像路径
        min_area: 最小区域面积，默认100
        max_area: 最大区域面积，默认10000
        min_aspect_ratio: 最小宽高比，默认0.1
        max_aspect_ratio: 最大宽高比，默认10.0
        output_path: 输出图像路径（绘制检测框），如果为None则不保存

    Returns:
        检测结果列表，每个元素包含:
        {
            'box': (x1, y1, x2, y2),
            'center': (cx, cy),
            'area': 区域面积
        }

    Raises:
        FileNotFoundError: 图像文件不存在
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取灰度图像
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 二值化（自适应阈值）
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=5,
    )

    # 形态学增强
    # 膨胀操作连接相邻文字
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)

    # 腐蚀操作去除噪点
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(dilated, kernel_erode, iterations=1)

    # 闭运算填充小孔
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    enhanced = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close)

    # 查找轮廓
    contours, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []

    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # 过滤太小或太大的区域
        if area < min_area or area > max_area:
            continue

        # 过滤异常宽高比
        aspect_ratio = w / max(h, 1)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        # 转换为 (x1, y1, x2, y2) 格式
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # 计算中心坐标
        center_x = float(x + w / 2)
        center_y = float(y + h / 2)

        regions.append(
            {
                "box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "area": area,
            }
        )

    # 如果指定了输出路径，绘制检测框并保存
    if output_path is not None:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for region in regions:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = region["center"]
            cv2.circle(output_image, (int(cx), int(cy)), 3, (255, 0, 0), -1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)

    return regions


def detect_single_chars_morphology(
    image_path: str | Path,
    char_size_range: tuple[int, int] = (15, 200),
    min_area_ratio: float = 0.2,
    separation_threshold: float = 1.5,
    output_path: str | Path | None = None,
) -> list[dict[str, tuple[int, int, int, int] | tuple[float, float]]]:
    """
    通过形态学操作检测单字级区域（针对古文字优化，支持字符分离）

    Args:
        image_path: 输入图像路径
        char_size_range: 字符大小范围 (min, max)，默认(15, 200)
        min_area_ratio: 最小面积比例（相对于min_size^2），默认0.2
        separation_threshold: 字符分离阈值（宽度超过平均宽度的倍数时进行分离），默认1.5
        output_path: 输出图像路径（绘制检测框），如果为None则不保存

    Returns:
        检测结果列表，每个元素包含:
        {
            'box': (x1, y1, x2, y2),
            'center': (cx, cy),
            'area': 区域面积
        }

    Raises:
        FileNotFoundError: 图像文件不存在
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取灰度图像
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    min_size, max_size = char_size_range

    # 第一步：检测字图（具有黑色底色的古文字图例）
    # 字图特征：有大片黑色背景，且是单个字符形式
    dark_threshold = 50  # 黑色阈值（灰度值低于此值认为是黑色）
    dark_mask = image < dark_threshold

    # 对黑色区域进行形态学操作，连接相邻的黑色像素
    kernel_dark = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel_dark, iterations=2)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_dark, iterations=1)

    # 查找黑色区域的轮廓
    dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 识别字图区域（具有大片黑色背景的区域）
    char_image_regions = []  # 存储字图的box，这些区域不应该被分割
    for contour in dark_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # 字图通常比普通文字大，且有明显的黑色背景
        # 检查区域内黑色像素的比例
        roi = image[y : y + h, x : x + w]
        dark_pixel_ratio = np.sum(roi < dark_threshold) / (w * h)

        # 字图特征：面积较大，黑色像素比例高，且宽高比接近正方形
        aspect_ratio = w / max(h, 1)
        if (
            area >= min_size * min_size * 2  # 面积至少是普通字符的2倍
            and area <= max_size * max_size * 4  # 但不超过最大尺寸的4倍
            and dark_pixel_ratio > 0.3  # 至少30%是黑色像素
            and 0.5 <= aspect_ratio <= 2.0
        ):  # 宽高比接近正方形
            char_image_regions.append((x, y, x + w, y + h))

    # 改进的二值化：尝试多种方法，选择最佳结果
    # 方法1：自适应阈值（更敏感，使用更小的blockSize）
    binary1 = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=9,  # 更小的blockSize，适应局部变化
        C=2,  # 进一步降低C值，提高敏感性
    )

    # 方法2：Otsu阈值
    _, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 方法3：更敏感的自适应阈值（用于捕获浅色文字）
    binary3 = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=1,  # 非常敏感
    )

    # 合并三种二值化结果（取并集，提高召回率）
    binary = cv2.bitwise_or(binary1, binary2)
    binary = cv2.bitwise_or(binary, binary3)

    # 形态学操作：去除噪点并预分离粘连字符
    # 对于黑底白字（二值化后字符为白色），先腐蚀再膨胀，减小粘连
    # 轻微腐蚀去除小噪点并拉开字符间距
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.erode(binary, kernel_erode, iterations=1)

    # 轻微膨胀恢复笔画但尽量不重新粘连
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)

    # 使用连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 第一遍：收集所有候选区域
    candidate_regions = []
    widths = []

    for i in range(1, num_labels):  # 跳过背景（label 0）
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # 放宽过滤条件，先收集所有候选
        if w < min_size or w > max_size * 2:  # 允许更大的宽度（可能是粘连字符）
            continue
        if h < min_size or h > max_size:
            continue

        # 放宽面积过滤
        if area < min_size * min_size * min_area_ratio:
            continue

        candidate_regions.append((x, y, w, h, area, cx, cy, i))
        if w <= max_size:  # 只统计正常大小的宽度
            widths.append(w)

    # 计算平均字符宽度（用于判断是否需要分离）
    avg_width = np.mean(widths) if widths else min_size * 2
    separation_width = avg_width * separation_threshold

    regions = []
    used_char_images = set()  # 记录已使用的字图索引，避免重复添加

    # 第二遍：处理每个候选区域，对粘连区域进行分离
    for x, y, w, h, area, cx, cy, label_id in candidate_regions:
        # 检查该区域是否与字图区域重叠，如果是字图，则不进行分割
        is_char_image = False
        box = (x, y, x + w, y + h)
        matched_char_image_idx = -1

        for idx, char_img_box in enumerate(char_image_regions):
            if idx in used_char_images:
                continue

            # 计算重叠度
            x1_1, y1_1, x2_1, y2_1 = box
            x1_2, y1_2, x2_2, y2_2 = char_img_box

            # 计算交集
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)

            if x2_i > x1_i and y2_i > y1_i:
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                box_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                overlap_ratio = intersection / box_area if box_area > 0 else 0

                # 如果重叠度超过50%，认为是字图的一部分
                if overlap_ratio > 0.5:
                    is_char_image = True
                    matched_char_image_idx = idx
                    break

        # 如果是字图，使用字图的完整box，而不是分割后的box
        if is_char_image and matched_char_image_idx not in used_char_images:
            char_img_box = char_image_regions[matched_char_image_idx]
            x1_2, y1_2, x2_2, y2_2 = char_img_box
            regions.append(
                {
                    "box": char_img_box,
                    "center": (float((x1_2 + x2_2) / 2), float((y1_2 + y2_2) / 2)),
                    "area": (x2_2 - x1_2) * (y2_2 - y1_2),
                }
            )
            used_char_images.add(matched_char_image_idx)
            continue

        # 如果宽度超过阈值，尝试分离字符
        if w > separation_width and w > min_size * 2:  # 只有明显粘连的才分离
            # 提取该区域的ROI（在原图上提取，用于投影分析）
            roi_x = max(0, x - 2)
            roi_y = max(0, y - 2)
            roi_w = min(w + 4, image.shape[1] - roi_x)
            roi_h = min(h + 4, image.shape[0] - roi_y)

            roi_binary = binary[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            # 使用垂直投影分析来分离字符
            vertical_projection = np.sum(roi_binary, axis=0)

            # 找到投影中的低谷（字符间隙）
            # 使用更智能的阈值：基于投影的统计特性
            proj_mean = np.mean(vertical_projection)
            proj_std = np.std(vertical_projection)
            threshold = max(proj_mean * 0.2, proj_mean - proj_std * 0.5)  # 更敏感的阈值

            gaps = []
            in_gap = False
            gap_start = 0

            for i, proj_val in enumerate(vertical_projection):
                if proj_val < threshold:
                    if not in_gap:
                        gap_start = i
                        in_gap = True
                else:
                    if in_gap:
                        gap_end = i
                        gap_width = gap_end - gap_start
                        # 只保留足够宽的间隙（避免噪点）
                        if gap_width >= 2:  # 降低最小间隙宽度
                            gaps.append((gap_start, gap_end))
                        in_gap = False

            # 如果有间隙，分割区域
            if gaps:
                split_positions = [0]
                for gap_start, gap_end in gaps:
                    # 分割点取间隙中间
                    split_pos = (gap_start + gap_end) // 2
                    split_positions.append(split_pos)
                split_positions.append(roi_w)

                # 为每个分割区域创建字符框
                for i in range(len(split_positions) - 1):
                    local_start = split_positions[i]
                    local_end = split_positions[i + 1]
                    char_w = local_end - local_start

                    # 检查这个区域是否有足够的像素（避免空白区域）
                    local_proj = vertical_projection[local_start:local_end]
                    if np.sum(local_proj) < threshold * char_w * 0.5:  # 空白区域
                        continue

                    if char_w >= min_size and char_w <= max_size:
                        char_x = x + local_start - 2  # 转换回原图坐标
                        if char_x < 0:
                            char_x = 0
                        char_y = y
                        char_h = h
                        char_area = char_w * char_h

                        if char_area >= min_size * min_size * min_area_ratio:
                            char_cx = char_x + char_w / 2
                            char_cy = char_y + char_h / 2

                            regions.append(
                                {
                                    "box": (char_x, char_y, char_x + char_w, char_y + char_h),
                                    "center": (float(char_cx), float(char_cy)),
                                    "area": char_area,
                                }
                            )

            # 如果分离失败或没有间隙，但宽度在合理范围内，仍然添加（可能是单个大字符）
            if not gaps and w <= max_size * 1.5:  # 放宽限制
                regions.append(
                    {
                        "box": (x, y, x + w, y + h),
                        "center": (float(cx), float(cy)),
                        "area": area,
                    }
                )
        else:
            # 正常大小的区域，直接添加
            regions.append(
                {
                    "box": (x, y, x + w, y + h),
                    "center": (float(cx), float(cy)),
                    "area": area,
                }
            )

    # 合并重叠的检测框
    if regions:
        # 分离字图区域和普通区域
        char_image_boxes = set(char_image_regions)
        char_image_regions_list = []
        normal_regions_list = []

        for region in regions:
            if region["box"] in char_image_boxes:
                char_image_regions_list.append(region)
            else:
                normal_regions_list.append(region)

        # 1. 先对字图区域进行合并（使用较低的阈值，避免过度合并）
        if char_image_regions_list:
            char_image_boxes_list = [region["box"] for region in char_image_regions_list]
            merged_char_image_boxes = merge_overlapping_boxes(
                char_image_boxes_list, overlap_threshold=0.5, containment_ratio=0.8
            )

            # 根据合并后的box重新构建字图区域
            merged_char_image_regions = []
            for box in merged_char_image_boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                area = w * h
                center_x = float(x1 + w / 2)
                center_y = float(y1 + h / 2)

                merged_char_image_regions.append(
                    {
                        "box": box,
                        "center": (center_x, center_y),
                        "area": area,
                    }
                )
        else:
            merged_char_image_regions = []

        # 2. 对普通区域进行合并
        if normal_regions_list:
            normal_boxes = [region["box"] for region in normal_regions_list]
            merged_boxes = merge_overlapping_boxes(normal_boxes, overlap_threshold=0.3)

            # 根据合并后的box重新构建结果
            merged_normal_regions = []
            for box in merged_boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                area = w * h
                center_x = float(x1 + w / 2)
                center_y = float(y1 + h / 2)

                merged_normal_regions.append(
                    {
                        "box": box,
                        "center": (center_x, center_y),
                        "area": area,
                    }
                )

            # 3. 移除与字图重叠的普通区域（优先保留字图）
            if merged_char_image_regions:
                char_image_boxes_set = {region["box"] for region in merged_char_image_regions}
                filtered_normal_regions = []

                for normal_region in merged_normal_regions:
                    normal_box = normal_region["box"]
                    should_keep = True

                    # 检查是否与任何字图重叠
                    for char_image_box in char_image_boxes_set:
                        # 计算重叠度
                        x1_1, y1_1, x2_1, y2_1 = normal_box
                        x1_2, y1_2, x2_2, y2_2 = char_image_box

                        # 计算交集
                        x1_i = max(x1_1, x1_2)
                        y1_i = max(y1_1, y1_2)
                        x2_i = min(x2_1, x2_2)
                        y2_i = min(y2_1, y2_2)

                        if x2_i > x1_i and y2_i > y1_i:
                            intersection = (x2_i - x1_i) * (y2_i - y1_i)
                            normal_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                            overlap_ratio = intersection / normal_area if normal_area > 0 else 0

                            # 如果重叠度超过30%，移除普通区域（优先保留字图）
                            if overlap_ratio > 0.3:
                                should_keep = False
                                break

                    if should_keep:
                        filtered_normal_regions.append(normal_region)

                merged_normal_regions = filtered_normal_regions
        else:
            merged_normal_regions = []

        # 4. 合并字图区域和普通区域
        regions = merged_char_image_regions + merged_normal_regions

    # 如果指定了输出路径，绘制检测框并保存
    if output_path is not None:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for region in regions:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = region["center"]
            cv2.circle(output_image, (int(cx), int(cy)), 3, (255, 0, 0), -1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)

    return regions


def detect_with_mser(
    image_path: str | Path,
    delta: int = 5,
    min_area: int = 100,
    max_area: int = 14400,
    max_variation: float = 0.25,
    min_size: int = 10,
    output_path: str | Path | None = None,
) -> list[dict[str, tuple[int, int, int, int] | tuple[float, float]]]:
    """
    使用MSER（最大稳定极值区域）检测文字区域

    Args:
        image_path: 输入图像路径
        delta: MSER delta参数，默认5
        min_area: 最小区域面积，默认100
        max_area: 最大区域面积，默认14400
        max_variation: 最大变化率，默认0.25
        min_size: 最小尺寸（宽或高），默认10
        output_path: 输出图像路径（绘制检测框），如果为None则不保存

    Returns:
        检测结果列表，每个元素包含:
        {
            'box': (x1, y1, x2, y2),
            'center': (cx, cy),
            'area': 区域面积
        }

    Raises:
        FileNotFoundError: 图像文件不存在
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取灰度图像
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 创建MSER检测器
    mser = cv2.MSER_create(
        delta=delta,
        min_area=min_area,
        max_area=max_area,
        max_variation=max_variation,
    )

    # 检测区域
    regions_mser, _ = mser.detectRegions(image)

    regions = []

    for region in regions_mser:
        x, y, w, h = cv2.boundingRect(region)

        # 过滤异常区域
        if w < min_size or h < min_size:
            continue

        # 转换为 (x1, y1, x2, y2) 格式
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # 计算中心坐标
        center_x = float(x + w / 2)
        center_y = float(y + h / 2)
        area = w * h

        regions.append(
            {
                "box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "area": area,
            }
        )

    # 如果指定了输出路径，绘制检测框并保存
    if output_path is not None:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for region in regions:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = region["center"]
            cv2.circle(output_image, (int(cx), int(cy)), 3, (255, 0, 0), -1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)

    return regions


def detect_abnormal_boxes(
    image_path: str | Path,
    boxes: list[tuple[int, int, int, int]] | None = None,
    corners: (list[tuple[int, int, int, int]] | list[list[tuple[int, int]]] | None) = None,
    outlier_method: str = "iqr",
    zscore_threshold: float = 2.5,
    output_path: str | Path | None = None,
) -> list[dict[str, tuple[int, int, int, int] | str]]:
    """
    检测并分类异常框（基于高度的异常值检测）

    Args:
        image_path: 输入图像路径
        boxes: box列表，每个box格式为 (x1, y1, x2, y2)
        corners: 对角角点列表，可以是：
            - list[tuple[int, int, int, int]]: [(x1, y1, x2, y2), ...]
            - list[list[tuple[int, int]]]: [[(x1, y1), (x2, y2)], ...]
        outlier_method: 异常值检测方法，'iqr'（四分位距）或'zscore'（Z分数），默认'iqr'
        zscore_threshold: Z分数阈值（仅当outlier_method='zscore'时使用），默认2.5
        output_path: 输出图像路径（绘制检测框和标签），如果为None则不保存

    Returns:
        检测结果列表，每个元素包含:
        {
            'box': (x1, y1, x2, y2),
            'tag': 'normal' 或 'abnormal',
            'row_index': 行号（从1开始），
            'col_index': 列号（从1开始，行内从左到右）
        }
        结果按行顺序排列（从上到下，每行内从左到右）

    Raises:
        FileNotFoundError: 图像文件不存在
        ValueError: 参数无效
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    if boxes is None and corners is None:
        raise ValueError("必须提供boxes或corners参数之一")

    # 读取图像（用于验证和可视化）
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 步骤1: 统一转换为box格式
    normalized_boxes = []

    if boxes is not None:
        if not isinstance(boxes, list):
            raise ValueError("boxes必须是列表")
        for box in boxes:
            if not isinstance(box, tuple) or len(box) != 4:
                raise ValueError(f"box格式无效: {box}，应为(x1, y1, x2, y2)")
            x1, y1, x2, y2 = box
            # 确保坐标顺序正确
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            normalized_boxes.append((x1, y1, x2, y2))

    if corners is not None:
        if not isinstance(corners, list):
            raise ValueError("corners必须是列表")
        for corner_item in corners:
            if isinstance(corner_item, tuple) and len(corner_item) == 4:
                # 格式: (x1, y1, x2, y2)
                x1, y1, x2, y2 = corner_item
            elif isinstance(corner_item, list) and len(corner_item) == 2:
                # 格式: [(x1, y1), (x2, y2)]
                (x1, y1), (x2, y2) = corner_item
            else:
                raise ValueError(f"corners格式无效: {corner_item}，应为(x1, y1, x2, y2)或[(x1, y1), (x2, y2)]")
            # 确保坐标顺序正确
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            normalized_boxes.append((x1, y1, x2, y2))

    if not normalized_boxes:
        return []

    # 步骤2: 按横向排布对所有box分组（按行分组）
    def is_same_row(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
        """
        判断两个box是否在同一行
        判断逻辑：
        1. box1的中心点y坐标处于box2的上下边缘区间中，或
        2. box1的上边缘或下边缘处于box2的上下边缘区间中，或
        3. box2的中心点y坐标处于box1的上下边缘区间中，或
        4. box2的上边缘或下边缘处于box1的上下边缘区间中
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算中心点和边缘
        cy1 = (y1_1 + y2_1) / 2
        cy2 = (y1_2 + y2_2) / 2

        # 判断条件1: box1的中心点y坐标处于box2的上下边缘区间中
        if y1_2 <= cy1 <= y2_2:
            return True

        # 判断条件2: box1的上边缘或下边缘处于box2的上下边缘区间中
        if y1_2 <= y1_1 <= y2_2 or y1_2 <= y2_1 <= y2_2:
            return True

        # 判断条件3: box2的中心点y坐标处于box1的上下边缘区间中
        if y1_1 <= cy2 <= y2_1:
            return True

        # 判断条件4: box2的上边缘或下边缘处于box1的上下边缘区间中
        if y1_1 <= y1_2 <= y2_1 or y1_1 <= y2_2 <= y2_1:
            return True

        return False

    # 行分组：遵循明确的遍历与标记步骤，避免顺序偏差
    n = len(normalized_boxes)
    marked = [False] * n  # 记录是否已分入某一行
    rows: list[list[tuple[int, int, int, int]]] = []

    def same_row_by_rules(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
        """严格按照规则判断是否同一行"""
        x1_1, y1_1, x2_1, y2_1 = b1
        x1_2, y1_2, x2_2, y2_2 = b2
        cy1 = (y1_1 + y2_1) / 2
        cy2 = (y1_2 + y2_2) / 2

        # 4. 中心点Y是否相同（允许整数比较，保持鲁棒性用近似相等）
        if int(round(cy1)) == int(round(cy2)):
            return True

        # 5. b1中心Y在b2上下边缘区间
        if y1_2 <= cy1 <= y2_2:
            return True

        # 5 对称：b2中心Y在b1上下边缘区间
        if y1_1 <= cy2 <= y2_1:
            return True

        # # 6. b1上/下边缘在b2上下边缘区间
        # if y1_2 <= y1_1 <= y2_2 or y1_2 <= y2_1 <= y2_2:
        #     return True

        # # 7. b2上/下边缘在b1上下边缘区间
        # if y1_1 <= y1_2 <= y2_1 or y1_1 <= y2_2 <= y2_1:
        #     return True

        return False

    # 2-8. 依次遍历，未标记则扩展所属行
    for i in range(n):
        if marked[i]:
            continue

        current_row = [normalized_boxes[i]]
        marked[i] = True
        changed = True

        while changed:
            changed = False
            for j in range(n):
                if marked[j]:
                    continue
                # 与当前行中任一box满足规则则入同一行
                if any(same_row_by_rules(row_box, normalized_boxes[j]) for row_box in current_row):
                    current_row.append(normalized_boxes[j])
                    marked[j] = True
                    changed = True

        # 9. 行内按中心点x排序
        current_row.sort(key=lambda b: (b[0] + b[2]) / 2)

        # 10. 对同一行内的box中心点进行直线拟合，判定斜率
        def fit_line_and_check_slope(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
            """
            对box中心点进行直线拟合，如果斜率不在-15°～15°之间，则拆分行

            Args:
                boxes: 同一行的box列表

            Returns:
                处理后的行列表（可能拆分成多行）
            """
            if len(boxes) <= 1:
                return [boxes]
            print(boxes)
            # 计算每个box的中心点
            centers = []
            for box in boxes:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append((center_x, center_y, box))

            # 如果只有2个点，直接计算斜率
            if len(centers) == 2:
                x1, y1, _ = centers[0]
                x2, y2, _ = centers[1]
                if abs(x2 - x1) < 1e-6:  # 垂直线
                    angle = 90.0
                else:
                    slope = (y2 - y1) / (x2 - x1)
                    angle = np.degrees(np.arctan(slope))
                print(f"角度: {angle:.2f}°")
                if -15 <= angle <= 15:
                    return [boxes]
                else:
                    # 按y从小到大排序，y小的单独成行
                    centers_sorted = sorted(centers, key=lambda c: c[1])
                    return [[centers_sorted[0][2]], [centers_sorted[1][2]]]

            # 使用最小二乘法拟合直线 y = ax + b
            x_coords = np.array([c[0] for c in centers])
            y_coords = np.array([c[1] for c in centers])

            # 如果x坐标都相同（垂直线），直接拆分
            if np.std(x_coords) < 1e-6:
                # 按y从小到大排序，逐个拆分
                centers_sorted = sorted(centers, key=lambda c: c[1])
                return [[c[2]] for c in centers_sorted]

            # 拟合直线
            coeffs = np.polyfit(x_coords, y_coords, 1)
            slope = coeffs[0]
            angle = np.degrees(np.arctan(slope))
            print(f"角度: {angle:.2f}°")
            # 如果斜率在-15°～15°之间，保持不变
            if -15 <= angle <= 15:
                return [boxes]

            # 如果不在范围内，按y从小到大去掉点后重新拟合
            centers_with_index = [(i, c) for i, c in enumerate(centers)]
            remaining_centers = sorted(centers_with_index, key=lambda item: item[1][1])  # 按y排序
            removed_boxes = []

            # 尝试去掉点直到斜率符合要求或只剩一个点
            while len(remaining_centers) > 1:
                # 提取当前剩余点的坐标
                current_x = np.array([item[1][0] for item in remaining_centers])
                current_y = np.array([item[1][1] for item in remaining_centers])

                # 如果x坐标都相同，无法拟合，直接拆分
                if np.std(current_x) < 1e-6:
                    break

                # 拟合直线
                coeffs = np.polyfit(current_x, current_y, 1)
                slope = coeffs[0]
                angle = np.degrees(np.arctan(slope))
                print(f"角度: {angle:.2f}°")
                # 如果斜率符合要求，停止
                if -15 <= angle <= 15:
                    break

                # 去掉y最小的点（第一个点）
                removed_item = remaining_centers.pop(0)
                removed_boxes.append(removed_item[1][2])  # 保存被移除的box

            # 构建结果：剩余的点组成一行，被移除的点各自成行
            result_rows = []
            if remaining_centers:
                # 剩余的点组成一行
                remaining_row = [item[1][2] for item in remaining_centers]
                result_rows.append(remaining_row)

            # 被移除的点各自成行
            for removed_box in removed_boxes:
                result_rows.append([removed_box])

            return result_rows

        # 对当前行进行直线拟合和拆分
        processed_rows = fit_line_and_check_slope(current_row)
        rows.extend(processed_rows)

    # 行按最小y从上到下排序，保持稳定输出
    rows.sort(key=lambda row: min(box[1] for box in row))

    # 步骤3: 对每行的box高度进行异常值查找
    def detect_outliers_iqr(heights: list[int]) -> set[int]:
        """使用IQR方法检测异常值"""
        if len(heights) < 3:
            # 如果少于3个box，不进行异常值检测
            return set()

        q1 = np.percentile(heights, 25)
        q3 = np.percentile(heights, 75)
        iqr = q3 - q1

        if iqr == 0:
            # 如果IQR为0，所有值相同，不标记异常
            return set()

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = set()
        for idx, height in enumerate(heights):
            if height < lower_bound or height > upper_bound:
                outliers.add(idx)

        return outliers

    def detect_outliers_zscore(heights: list[int], threshold: float = 2.5) -> set[int]:
        """使用Z分数方法检测异常值"""
        if len(heights) < 3:
            # 如果少于3个box，不进行异常值检测
            return set()

        mean = np.mean(heights)
        std = np.std(heights)

        if std == 0:
            # 如果标准差为0，所有值相同，不标记异常
            return set()

        outliers = set()
        for idx, height in enumerate(heights):
            z_score = abs((height - mean) / std)
            if z_score > threshold:
                outliers.add(idx)

        return outliers

    # 步骤4: 按行顺序输出所有box，带上tag、行号和列号
    # 注意：tag统一设置为"normal"，后续会结合OCR结果进行异常判断
    result = []
    for row_idx, row in enumerate(rows, start=1):
        # 为每个box添加tag、行号和列号（tag统一为normal）
        for col_idx, box in enumerate(row, start=1):
            result.append(
                {
                    "box": box,
                    "tag": "normal",  # 统一设置为normal，后续结合OCR结果判断
                    "row_index": row_idx,
                    "col_index": col_idx,
                }
            )

    # 如果指定了输出路径，绘制检测框和标签
    if output_path is not None:
        output_image = image.copy()
        h, w = output_image.shape[:2]

        # 按行分组结果，用于绘制拟合直线
        rows_dict: dict[int, list[dict]] = {}
        for item in result:
            row_idx = item["row_index"]
            if row_idx not in rows_dict:
                rows_dict[row_idx] = []
            rows_dict[row_idx].append(item)

        # 绘制每行的拟合直线
        for row_idx, row_items in rows_dict.items():
            if len(row_items) < 2:
                continue  # 少于2个box无法拟合直线

            # 计算每个box的中心点
            centers = []
            for item in row_items:
                x1, y1, x2, y2 = item["box"]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append((center_x, center_y))

            x_coords = np.array([c[0] for c in centers])
            y_coords = np.array([c[1] for c in centers])

            # 如果x坐标都相同（垂直线），跳过
            if np.std(x_coords) < 1e-6:
                continue

            # 拟合直线 y = ax + b
            try:
                coeffs = np.polyfit(x_coords, y_coords, 1)
                slope = coeffs[0]
                intercept = coeffs[1]

                # 计算直线在图像范围内的两个端点
                # 从x=0到x=w
                x_start = 0
                y_start = int(slope * x_start + intercept)
                x_end = w - 1
                y_end = int(slope * x_end + intercept)

                # 如果直线超出图像范围，调整端点
                if y_start < 0:
                    y_start = 0
                    x_start = int((y_start - intercept) / slope) if abs(slope) > 1e-6 else 0
                elif y_start >= h:
                    y_start = h - 1
                    x_start = int((y_start - intercept) / slope) if abs(slope) > 1e-6 else 0

                if y_end < 0:
                    y_end = 0
                    x_end = int((y_end - intercept) / slope) if abs(slope) > 1e-6 else w - 1
                elif y_end >= h:
                    y_end = h - 1
                    x_end = int((y_end - intercept) / slope) if abs(slope) > 1e-6 else w - 1

                # 确保坐标在图像范围内
                x_start = max(0, min(x_start, w - 1))
                y_start = max(0, min(y_start, h - 1))
                x_end = max(0, min(x_end, w - 1))
                y_end = max(0, min(y_end, h - 1))

                # 绘制拟合直线（使用蓝色，线宽1）
                cv2.line(output_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1, cv2.LINE_AA)
            except Exception:
                # 如果拟合失败，跳过
                pass

        # 绘制检测框和标签
        for item in result:
            x1, y1, x2, y2 = item["box"]
            tag = item["tag"]

            # 根据tag选择颜色：normal=绿色，abnormal=红色
            color = (0, 255, 0) if tag == "normal" else (0, 0, 255)
            thickness = 2

            # 绘制矩形框
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签文本
            label_text = tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

            # 在框上方绘制文本背景
            text_x = x1
            text_y = max(y1 - 5, text_height + 5)
            cv2.rectangle(
                output_image,
                (text_x, text_y - text_height - 5),
                (text_x + text_width, text_y + baseline),
                color,
                -1,
            )

            # 绘制文本
            cv2.putText(
                output_image,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),  # 白色文本
                text_thickness,
                cv2.LINE_AA,
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)
    # print(result)
    return result
