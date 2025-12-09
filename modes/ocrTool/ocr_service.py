"""OCR服务模块"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

from modes.imageTool import image_service


class OCRService:
    """OCR服务类，封装RapidOCR功能"""

    def __init__(self):
        """
        初始化OCR引擎

        注意：根据rapidocr-onnxruntime的实际API，参数配置可能需要通过配置文件
        或使用默认配置。如果您的版本支持params参数，可以取消注释下面的代码。
        """
        # 使用默认配置初始化（适用于大多数情况）
        self.ocr_engine = RapidOCR(
            params={
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Global.max_side_len": 2000,
            },
        )

    def detect_characters(
        self, image_path: str | Path, max_side: int = 2000
    ) -> tuple[List[Tuple[List[List[float]], str, float]], np.ndarray, float]:
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
        resized_image, scale = image_service.resize_image_by_max_side(
            image_path, max_side=max_side
        )

        # 保存缩放后的图像到临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            cv2.imwrite(str(tmp_path), resized_image)

        try:
            # 使用单字级别OCR (return_word_box=True) 对缩放后的图像进行OCR
            ocr_result = self.ocr_engine(str(tmp_path), return_word_box=True)
        finally:
            # 清理临时文件
            tmp_path.unlink(missing_ok=True)

        # 处理单字级别结果
        word_results = []
        if hasattr(ocr_result, "word_results"):
            # word_results是嵌套列表: [[[box, text, score], ...], ...]
            # 每个外层列表是一行，每个内层列表包含单词结果
            for line_results in ocr_result.word_results:
                if not line_results:
                    continue
                for word_result in line_results:
                    if (
                        not isinstance(word_result, (list, tuple))
                        or len(word_result) < 3
                    ):
                        continue

                    # 识别box、text和confidence
                    dt_box = None
                    text = None
                    confidence = None

                    # 尝试通过类型识别
                    for item in word_result:
                        if isinstance(item, (list, tuple)) and len(item) >= 4:
                            # 检查是否是box（坐标点列表）
                            if all(
                                isinstance(p, (list, tuple)) and len(p) >= 2
                                for p in item
                            ):
                                try:
                                    _ = [float(p[0]) for p in item]
                                    _ = [float(p[1]) for p in item]
                                    dt_box = item
                                except (ValueError, TypeError, IndexError):
                                    pass
                        elif isinstance(item, str):
                            text = item
                        elif isinstance(item, (int, float)) or (
                            hasattr(np, "number") and isinstance(item, np.number)
                        ):
                            confidence = item

                    # 如果还没找到，尝试默认顺序
                    if dt_box is None or text is None or confidence is None:
                        if (
                            isinstance(word_result[0], (list, tuple))
                            and len(word_result[0]) >= 4
                        ):
                            dt_box = word_result[0]
                        elif (
                            isinstance(word_result[1], (list, tuple))
                            and len(word_result[1]) >= 4
                        ):
                            dt_box = word_result[1]
                        elif (
                            isinstance(word_result[2], (list, tuple))
                            and len(word_result[2]) >= 4
                        ):
                            dt_box = word_result[2]

                        if isinstance(word_result[0], str):
                            text = word_result[0]
                        elif isinstance(word_result[1], str):
                            text = word_result[1]
                        elif isinstance(word_result[2], str):
                            text = word_result[2]

                        if isinstance(word_result[0], (int, float)) or (
                            hasattr(np, "number")
                            and isinstance(word_result[0], np.number)
                        ):
                            confidence = word_result[0]
                        elif isinstance(word_result[1], (int, float)) or (
                            hasattr(np, "number")
                            and isinstance(word_result[1], np.number)
                        ):
                            confidence = word_result[1]
                        elif isinstance(word_result[2], (int, float)) or (
                            hasattr(np, "number")
                            and isinstance(word_result[2], np.number)
                        ):
                            confidence = word_result[2]

                    # 验证所有必需组件
                    if dt_box is None or text is None or confidence is None:
                        continue

                    word_results.append((dt_box, text, float(confidence)))

        # 如果没有word_results，回退到行级别OCR
        if not word_results:
            # 行级别OCR
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                cv2.imwrite(str(tmp_path), resized_image)

            try:
                ocr_result = self.ocr_engine(str(tmp_path))

                # 处理不同的返回格式
                if hasattr(ocr_result, "boxes"):
                    # 新格式: RapidOCROutput对象
                    result = list(
                        zip(ocr_result.boxes, ocr_result.txts, ocr_result.scores)
                    )
                else:
                    # 旧格式: tuple (result, _)
                    result, _ = ocr_result

                for item in result:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        dt_box = item[0]
                        text = item[1]
                        confidence = item[2]
                        word_results.append((dt_box, text, float(confidence)))
            finally:
                tmp_path.unlink(missing_ok=True)

        return word_results, resized_image, scale

    def recognize_character(
        self, image: np.ndarray, box: List[List[float]]
    ) -> Tuple[str, float]:
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

    def extract_character_images(
        self,
        image_path: str | Path,
        output_dir: Optional[str | Path] = None,
        save_images: bool = True,
        max_side: int = 2000,
    ) -> tuple[List[Dict[str, Any]], np.ndarray, float]:
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
                image_filename = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in image_filename
                )
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
        output_dir: Optional[str | Path] = None,
        save_character_images: bool = True,
        max_side: int = 2000,
    ) -> Dict[str, Any]:
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


# 创建全局OCR服务实例
_ocr_service_instance: Optional[OCRService] = None


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
) -> List[Tuple[List[List[float]], str, float]]:
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
    output_dir: Optional[str | Path] = None,
    save_images: bool = True,
    max_side: int = 2000,
) -> tuple[List[Dict[str, Any]], np.ndarray, float]:
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
    return get_ocr_service().extract_character_images(
        image_path, output_dir, save_images, max_side
    )


def process_image(
    image_path: str | Path,
    output_dir: Optional[str | Path] = None,
    save_character_images: bool = True,
    max_side: int = 2000,
) -> Dict[str, Any]:
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
    return get_ocr_service().process_image(
        image_path, output_dir, save_character_images, max_side
    )
