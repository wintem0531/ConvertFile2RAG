"""OCR服务模块"""

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr import RapidOCR

from modes.imageTool import image_service


class OCRService:
    """OCR服务类，封装RapidOCR功能"""

    def __init__(self, config_path: str | Path | None = None):
        """
        初始化OCR引擎

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径 modes/ocrTool/config.yaml
        """
        # 确定配置文件路径
        if config_path is None:
            # 获取当前文件所在目录
            current_dir = Path(__file__).parent
            config_path = current_dir / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"OCR配置文件不存在: {config_path}")

        # 直接使用配置文件路径初始化OCR引擎
        # RapidOCR支持直接传入config_path参数
        self.ocr_engine = RapidOCR(config_path=str(config_path))

    def detect_characters(
        self, image_path: str | Path, max_side: int = 2000
    ) -> tuple[list[tuple[list[list[float]], str, float]], np.ndarray, float]:
        """
        使用OCR的det步骤，获取图像中所有"单个字"的定位框

        Args:
            image_path: 输入图像路径
            max_side: 最长边的最大像素值，默认2000

        Returns:
            (检测结果列表, 缩放后的图像, 缩放比例)
            检测结果列表每个元素为 (box坐标, 识别文本, 置信度)
            box格式: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 先缩放图像至最长边不超过max_side
        resized_image, scale = image_service.resize_image_by_max_side(image_path, max_side=max_side)

        # 保存缩放后的图像到临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            cv2.imwrite(str(tmp_path), resized_image)

        try:
            # 使用标准OCR对缩放后的图像进行OCR
            ocr_result = self.ocr_engine(str(tmp_path))
        finally:
            # 清理临时文件
            tmp_path.unlink(missing_ok=True)

        # 处理OCR结果
        word_results = []

        # 处理不同的返回格式
        if hasattr(ocr_result, "boxes"):
            # 新格式: RapidOCROutput对象
            result = list(zip(ocr_result.boxes, ocr_result.txts, ocr_result.scores))
        else:
            # 旧格式: tuple (result, _)
            result, _ = ocr_result

        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                dt_box = item[0]
                text = item[1]
                confidence = item[2]
                word_results.append((dt_box, text, float(confidence)))

        return word_results, resized_image, scale

    def recognize_character(self, image: np.ndarray, box: list[list[float]]) -> tuple[str, float]:
        """
        对单个字符进行OCR识别

        Args:
            image: 原始图像numpy数组
            box: 字符定位框，格式 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        Returns:
            (识别文本, 置信度)
        """
        # 裁剪出字符区域
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # 确保坐标在图像范围内
        h, w = image.shape[:2]
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(x_min + 1, min(x_max, w))
        y_max = max(y_min + 1, min(y_max, h))

        char_image = image[y_min:y_max, x_min:x_max]

        # 对裁剪出的字符图像进行识别
        if char_image.size == 0:
            return "", 0.0

        # RapidOCR可以接受numpy数组
        ocr_result = self.ocr_engine(char_image)

        # 处理不同的返回格式
        if hasattr(ocr_result, "boxes"):
            # 新格式: RapidOCROutput对象
            if len(ocr_result.boxes) > 0:
                text = ocr_result.txts[0] if ocr_result.txts else ""
                confidence = float(ocr_result.scores[0]) if ocr_result.scores else 0.0
                return text, confidence
        else:
            # 旧格式: tuple (result, _)
            result, _ = ocr_result
            if result and len(result) > 0:
                _, text, confidence = result[0]
                return text, float(confidence)

        return "", 0.0

    def recognize_text_only(self, image: np.ndarray | str | Path) -> tuple[str, float, list[list[list[float]]] | None]:
        """
        只进行分类（cls）和识别（rec），不进行检测（det）
        用于识别已经裁剪好的图像区域

        Args:
            image: 输入图像，可以是numpy数组、文件路径或Path对象

        Returns:
            (识别文本, 置信度, 单字坐标列表)
            如果没有识别到文字，返回 ("", 0.0, None)
            单字坐标格式: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
        """
        # 创建只开启cls和rec的OCR引擎
        cls_rec_engine = RapidOCR(
            config_path=str(Path(__file__).parent / "config.yaml"),
            params={
                "Global.use_det": False,
                "Global.use_cls": True,
                "Global.use_rec": True,
                "Global.return_word_box": True,
            },
        )

        # 处理输入：如果是路径，读取图像；如果是numpy数组，保存到临时文件
        is_temp_file = False
        tmp_path = None

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            input_image = str(image_path)
        else:
            # numpy数组，保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                cv2.imwrite(str(tmp_path), image)
                input_image = str(tmp_path)
                is_temp_file = True

        try:
            # 使用只开启cls和rec的OCR引擎进行识别
            ocr_result = cls_rec_engine(input_image)
        finally:
            # 如果是临时文件，清理
            if is_temp_file and tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        # print(ocr_result)
        # 处理OCR结果
        text = ""
        confidence = 0.0
        word_boxes: list[list[list[float]]] | None = None

        # 处理不同的返回格式
        # 检查是否有 txts 属性（新格式: RapidOCROutput 或 TextRecOutput 对象）
        if hasattr(ocr_result, "txts"):
            if ocr_result.txts and len(ocr_result.txts) > 0:
                text = str(ocr_result.txts[0]) if ocr_result.txts[0] else ""
                if hasattr(ocr_result, "scores") and ocr_result.scores and len(ocr_result.scores) > 0:
                    confidence = float(ocr_result.scores[0])
            # 提取单字坐标（word_results）- 当 return_word_box=True 时，word_results 包含每个字符的坐标
            if hasattr(ocr_result, "word_results") and ocr_result.word_results is not None:
                try:
                    boxes_list = []
                    # word_results 是一个元组，每个元素是 (字符, 置信度, 坐标框)
                    for word_item in ocr_result.word_results:
                        if isinstance(word_item, (list, tuple)) and len(word_item) >= 3:
                            # 第三个元素是坐标框 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                            box = word_item[2]
                            if isinstance(box, (list, tuple)) and len(box) == 4:
                                # 验证每个点都是 [x, y] 格式
                                if all(isinstance(point, (list, tuple)) and len(point) == 2 for point in box):
                                    # 转换为列表格式，确保坐标是数字类型
                                    box_list = [[float(point[0]), float(point[1])] for point in box]
                                    boxes_list.append(box_list)
                    if boxes_list:
                        word_boxes = boxes_list
                except Exception:
                    word_boxes = None
        # 检查是否是 tuple 格式（旧格式）
        elif isinstance(ocr_result, tuple) and len(ocr_result) >= 2:
            result, _ = ocr_result
            if result and len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, (list, tuple)) and len(first_item) >= 3:
                    text = str(first_item[1]) if first_item[1] else ""
                    confidence = float(first_item[2]) if first_item[2] else 0.0
                    # 提取所有字符的boxes（当 return_word_box=True 时）
                    if len(result) > 0:
                        boxes_list = []
                        for item in result:
                            if isinstance(item, (list, tuple)) and len(item) > 0:
                                box = item[0]
                                if isinstance(box, (list, tuple)) and len(box) == 4:
                                    if all(isinstance(point, (list, tuple)) and len(point) == 2 for point in box):
                                        boxes_list.append(box)
                        if boxes_list:
                            word_boxes = boxes_list
        # 检查是否是列表格式
        elif isinstance(ocr_result, list) and len(ocr_result) > 0:
            first_item = ocr_result[0]
            if isinstance(first_item, (list, tuple)) and len(first_item) >= 3:
                text = str(first_item[1]) if first_item[1] else ""
                confidence = float(first_item[2]) if first_item[2] else 0.0
                # 提取所有字符的boxes（当 return_word_box=True 时）
                boxes_list = []
                for item in ocr_result:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        box = item[0]
                        if isinstance(box, (list, tuple)) and len(box) == 4:
                            if all(isinstance(point, (list, tuple)) and len(point) == 2 for point in box):
                                boxes_list.append(box)
                if boxes_list:
                    word_boxes = boxes_list

        return text, confidence, word_boxes

    def detect_only(
        self, image_path: str | Path, max_side: int = 2000
    ) -> tuple[list[list[list[float]]], np.ndarray, float]:
        """
        只进行检测（det），不进行分类（cls）和识别（rec）
        返回所有检测到的文本框坐标

        Args:
            image_path: 输入图像路径
            max_side: 最长边的最大像素值，默认2000

        Returns:
            (检测框列表, 缩放后的图像, 缩放比例)
            检测框格式: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 先缩放图像至最长边不超过max_side
        resized_image, scale = image_service.resize_image_by_max_side(image_path, max_side=max_side)

        # 创建只开启det的OCR引擎
        # 使用params参数覆盖配置，只开启det
        det_only_engine = RapidOCR(
            config_path=str(Path(__file__).parent / "config.yaml"),
            params={
                "Global.use_det": True,
                "Global.use_cls": False,
                "Global.use_rec": False,
                "Global.return_word_box": True,
            },
        )

        # 保存缩放后的图像到临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            cv2.imwrite(str(tmp_path), resized_image)

        try:
            # 使用只开启det的OCR引擎进行检测
            ocr_result = det_only_engine(str(tmp_path))
            # print(ocr_result)
        finally:
            # 清理临时文件
            tmp_path.unlink(missing_ok=True)

        # 提取检测框
        boxes = []

        # 处理不同的返回格式
        if hasattr(ocr_result, "boxes"):
            # 新格式: RapidOCROutput对象
            if ocr_result.boxes is not None and len(ocr_result.boxes) > 0:
                boxes = ocr_result.boxes.tolist() if hasattr(ocr_result.boxes, "tolist") else list(ocr_result.boxes)
            else:
                boxes = []
        else:
            # 旧格式: tuple (result, _)
            result, _ = ocr_result
            if result:
                # 提取所有检测框
                boxes = [item[0] for item in result if isinstance(item, (list, tuple)) and len(item) >= 1]

        return boxes, resized_image, scale

    def extract_character_images(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        save_images: bool = True,
        max_side: int = 2000,
    ) -> tuple[list[dict[str, Any]], np.ndarray, float]:
        """
        提取图像中所有单个字符的图像

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录，如果为None则不保存图像
            save_images: 是否保存字符图像，默认True
            max_side: 最长边的最大像素值，默认2000

        Returns:
            (字符信息列表, 缩放后的图像, 缩放比例)
            字符信息列表每个元素包含:
            {
                'box': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                'text': '识别文本',
                'confidence': 置信度,
                'image': numpy数组 (如果save_images=False),
                'image_path': 保存路径 (如果save_images=True)
            }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 检测所有字符（会自动缩放图像）
        det_results, resized_image, scale = self.detect_characters(image_path, max_side)

        output_dir = Path(output_dir) if output_dir else None
        if save_images and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        character_results = []

        for idx, (box, text, confidence) in enumerate(det_results):
            # 提取字符图像
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # 确保坐标在图像范围内（使用缩放后的图像）
            h, w = resized_image.shape[:2]
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))

            char_image = resized_image[y_min:y_max, x_min:x_max]

            result_dict = {
                "box": box,
                "text": text,
                "confidence": float(confidence),
            }

            if save_images and output_dir:
                # 保存字符图像
                image_filename = f"char_{idx:04d}_{text}_{confidence:.2f}.png"
                # 清理文件名中的非法字符
                image_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in image_filename)
                image_path_save = output_dir / image_filename
                cv2.imwrite(str(image_path_save), char_image)
                result_dict["image_path"] = str(image_path_save)
            else:
                result_dict["image"] = char_image

            character_results.append(result_dict)

        return character_results, resized_image, scale

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        save_character_images: bool = True,
        max_side: int = 2000,
    ) -> dict[str, Any]:
        """
        完整的OCR处理流程：检测字符 -> 识别 -> 提取字符图像

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录，用于保存字符图像
            save_character_images: 是否保存字符图像，默认True
            max_side: 最长边的最大像素值，默认2000

        Returns:
            处理结果字典，包含:
            {
                'image_path': 输入图像路径,
                'resized_image': 缩放后的图像numpy数组,
                'scale': 缩放比例,
                'characters': [
                    {
                        'box': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                        'text': '识别文本',
                        'confidence': 置信度,
                        'image_path': 字符图像路径 (如果保存)
                    },
                    ...
                ],
                'total_characters': 字符总数
            }
        """
        image_path = Path(image_path)

        # 提取所有字符图像（会自动缩放图像）
        character_results, resized_image, scale = self.extract_character_images(
            image_path, output_dir, save_character_images, max_side
        )

        return {
            "image_path": str(image_path),
            "resized_image": resized_image,
            "scale": scale,
            "characters": character_results,
            "total_characters": len(character_results),
        }

    def generate_character_image(
        self, character: str, size: tuple[int, int] = (32, 32), font_path: str | Path | None = None
    ) -> np.ndarray:
        """
        生成汉字对应的图片（白底黑字）

        Args:
            character: 输入的汉字（单个字符）
            size: 图片大小，默认(32, 32)
            font_path: 字体文件路径，如果为None则使用系统默认字体

        Returns:
            生成的图片numpy数组，格式为 (H, W, 3) 的BGR格式（与OpenCV兼容）

        Raises:
            ValueError: 输入不是单个字符
        """
        if len(character) != 1:
            raise ValueError(f"输入必须是单个字符，当前输入: {character}")

        width, height = size

        # 创建白色背景图像
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        # 尝试加载字体
        font = None
        if font_path:
            font_path = Path(font_path)
            if font_path.exists():
                try:
                    # 根据图片大小调整字体大小
                    font_size = min(width, height) - 4  # 留出边距
                    font = ImageFont.truetype(str(font_path), font_size)
                except Exception:
                    # 如果加载字体失败，使用默认字体
                    font = None

        # 如果没有指定字体或加载失败，使用默认字体
        if font is None:
            try:
                # 尝试使用系统默认的中文字体
                # 在macOS上使用PingFang SC，在Linux上使用WenQuanYi Micro Hei，在Windows上使用SimHei
                import platform

                system = platform.system()
                if system == "Darwin":  # macOS
                    font_paths = [
                        "/System/Library/Fonts/PingFang.ttc",
                        "/System/Library/Fonts/STHeiti Light.ttc",
                    ]
                elif system == "Linux":
                    font_paths = [
                        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                    ]
                else:  # Windows
                    font_paths = [
                        "C:/Windows/Fonts/simhei.ttf",
                        "C:/Windows/Fonts/msyh.ttc",
                    ]

                font_size = min(width, height) - 4
                for fp in font_paths:
                    if Path(fp).exists():
                        try:
                            font = ImageFont.truetype(fp, font_size)
                            break
                        except Exception:
                            continue
            except Exception:
                pass

        # 如果仍然没有字体，使用默认字体（可能不支持中文）
        if font is None:
            font_size = min(width, height) - 4
            try:
                font = ImageFont.load_default()
            except Exception:
                # 如果连默认字体都加载失败，使用None（PIL会使用内置字体）
                font = None

        # 计算文字位置（居中）
        if font:
            # 获取文字尺寸（兼容新旧PIL版本）
            try:
                # 新版本PIL使用textbbox
                bbox = draw.textbbox((0, 0), character, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # 旧版本PIL使用textsize
                try:
                    text_width, text_height = draw.textsize(character, font=font)
                except AttributeError:
                    # 如果都不可用，使用估算值
                    text_width = width * 0.8
                    text_height = height * 0.8
        else:
            # 如果没有字体，使用估算值
            text_width = width * 0.8
            text_height = height * 0.8

        # 计算居中位置
        x = (width - text_width) / 2
        y = (height - text_height) / 2

        # 绘制黑色文字
        draw.text((x, y), character, fill=(0, 0, 0), font=font)

        # 转换为numpy数组（RGB格式）
        image_array = np.array(image)

        # 转换为BGR格式（OpenCV使用BGR）
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return image_bgr


# 创建全局OCR服务实例
_ocr_service_instance: OCRService | None = None


def get_ocr_service() -> OCRService:
    """
    获取OCR服务实例（单例模式）

    Returns:
        OCRService实例
    """
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = OCRService()
    return _ocr_service_instance


# 便捷函数
def detect_characters(
    image_path: str | Path,
) -> list[tuple[list[list[float]], str, float]]:
    """
    便捷函数：检测图像中的字符

    Args:
        image_path: 输入图像路径

    Returns:
        检测结果列表
    """
    return get_ocr_service().detect_characters(image_path)


def extract_character_images(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    save_images: bool = True,
    max_side: int = 2000,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    """
    便捷函数：提取字符图像

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        save_images: 是否保存图像
        max_side: 最长边的最大像素值，默认2000

    Returns:
        (字符信息列表, 缩放后的图像, 缩放比例)
    """
    return get_ocr_service().extract_character_images(image_path, output_dir, save_images, max_side)


def process_image(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    save_character_images: bool = True,
    max_side: int = 2000,
) -> dict[str, Any]:
    """
    便捷函数：完整的OCR处理流程

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        save_character_images: 是否保存字符图像
        max_side: 最长边的最大像素值，默认2000

    Returns:
        处理结果字典
    """
    return get_ocr_service().process_image(image_path, output_dir, save_character_images, max_side)


def detect_only(
    image_path: str | Path,
    max_side: int = 2000,
) -> tuple[list[list[list[float]]], np.ndarray, float]:
    """
    便捷函数：只进行检测（det），不进行分类和识别

    Args:
        image_path: 输入图像路径
        max_side: 最长边的最大像素值，默认2000

    Returns:
        (检测框列表, 缩放后的图像, 缩放比例)
    """
    return get_ocr_service().detect_only(image_path, max_side)


def recognize_text_only(
    image: np.ndarray | str | Path,
) -> tuple[str, float, list[list[list[float]]] | None]:
    """
    便捷函数：只进行分类（cls）和识别（rec），不进行检测（det）

    Args:
        image: 输入图像，可以是numpy数组、文件路径或Path对象

    Returns:
        (识别文本, 置信度, 单字坐标列表)
        单字坐标格式: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
    """
    return get_ocr_service().recognize_text_only(image)


def generate_character_image(
    character: str,
    size: tuple[int, int] = (32, 32),
    font_path: str | Path | None = None,
) -> np.ndarray:
    """
    便捷函数：生成汉字对应的图片（白底黑字）

    Args:
        character: 输入的汉字（单个字符）
        size: 图片大小，默认(32, 32)
        font_path: 字体文件路径，如果为None则使用系统默认字体

    Returns:
        生成的图片numpy数组，格式为 (H, W, 3) 的BGR格式（与OpenCV兼容）

    Raises:
        ValueError: 输入不是单个字符
    """
    return get_ocr_service().generate_character_image(character, size, font_path)
