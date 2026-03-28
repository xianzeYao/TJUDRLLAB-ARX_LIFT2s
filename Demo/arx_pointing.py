"""
基于 OpenAI/vLLM 兼容接口的单点提取工具：
- 输入: 一张 RGB/BGR 图像 (np.ndarray)
- 输出: 第一组像素坐标 (u, v)

依赖:
- openai (兼容 vLLM)
- numpy
- opencv-python
"""
from __future__ import annotations

import ast
import base64
import re
from typing import Any, Dict, List, Sequence, Tuple, Optional, Union

import cv2
import numpy as np
from openai import OpenAI


# ---- 文本解析：从模型输出中提取二维点 ----
def _preprocess_text(text: str) -> str:
    text = re.sub(r"```(?:json|python|html)?\n?(.*?)\n?```",
                  r"\1", text, flags=re.DOTALL)
    tag_match = re.search(r"<(?:point|points)>(.*?)</(?:point|points)>",
                          text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        text = tag_match.group(1)
    return text.strip()


def _parse_structured_data(data: Any) -> List[List[float]]:
    points: List[List[float]] = []
    if isinstance(data, dict):
        for key in ["point_2d", "points", "point", "coordinates"]:
            if key in data:
                return _parse_structured_data(data[key])
    elif isinstance(data, (list, tuple)):
        if not data:
            return []
        if len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
            return [[float(data[0]), float(data[1])]]
        for item in data:
            extracted = _parse_structured_data(item)
            if extracted:
                points.extend(extracted)
    return points


def _extract_points_by_regex(text: str) -> List[List[float]]:
    points: List[List[float]] = []
    bracket_pattern = r"[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]"
    matches = re.findall(bracket_pattern, text)
    if matches:
        for m in matches:
            points.append([float(m[0]), float(m[1])])
    else:
        raw_pattern = r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)"
        matches = re.findall(raw_pattern, text)
        for m in matches:
            points.append([float(m[0]), float(m[1])])
    return points


def omni_decode_points(output: str) -> List[List[float]]:
    """鲁棒解析模型文本中的二维点列表。"""
    if not isinstance(output, str) or not output.strip():
        return []
    if "<point" in output.lower():
        pts: List[List[float]] = []
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', output):
            pts.append([float(match.group(1)), float(match.group(2))])
        if pts:
            return pts
    text = _preprocess_text(output)
    try:
        clean_text = re.sub(r"^[a-zA-Z0-9_\s]+:\s*", "", text)
        data = ast.literal_eval(clean_text)
        parsed = _parse_structured_data(data)
        if parsed:
            return parsed
    except (ValueError, SyntaxError, MemoryError):
        pass
    return _extract_points_by_regex(text)


# ---- 图像编码 ----
def _image_to_data_uri(image: np.ndarray, assume_bgr: bool = True) -> str:
    if image is None or image.ndim != 3:
        raise ValueError("image 需要为 HxWx3 的 numpy 数组")
    img = image
    if assume_bgr:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("图像编码失败")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _normalize_assume_bgr(
    assume_bgr: Union[bool, Sequence[bool]],
    image_count: int,
) -> List[bool]:
    if isinstance(assume_bgr, bool):
        return [assume_bgr] * image_count
    flags = list(assume_bgr)
    if len(flags) != image_count:
        raise ValueError(
            f"assume_bgr length mismatch: need {image_count}, got {len(flags)}"
        )
    return [bool(flag) for flag in flags]


def _build_point_prompt(text_prompt: str, all_prompt: str = "") -> str:
    if all_prompt:
        return all_prompt
    return (
        "Provide one or more points coordinate of objects region this sentence describes: "
        f"{text_prompt}. "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].'
    )


def _scale_points_to_image(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
    norm_range: float = 1000.0,
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for p in points:
        arr = np.array(p, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 2:
            raise RuntimeError(f"坐标维度异常: {arr}")

        if norm_range and np.max(np.abs(arr)) <= norm_range:
            arr = arr / norm_range * np.array([width, height], dtype=np.float64)

        u = float(np.clip(arr[0], 0, width - 1))
        v = float(np.clip(arr[1], 0, height - 1))
        result.append((u, v))
    return result


def predict_multi_points_from_multi_image(
    images: Sequence[np.ndarray],
    text_prompt: str,
    all_prompt: str = "",
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    assume_bgr: Union[bool, Sequence[bool]] = True,
    norm_range: float = 1000.0,
    temperature: float = 0.7,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 512,
    return_raw: bool = False,
) -> Union[List[Tuple[float, float]], Tuple[List[Tuple[float, float]], Optional[str]]]:
    """
    发送多张图像到 vLLM/OpenAI 兼容接口，返回解析后的像素点列表。
    点坐标按第一张图的尺寸做缩放与裁剪。
    """
    if not images:
        raise ValueError("images 不能为空")
    first_image = images[0]
    if first_image is None or first_image.ndim != 3:
        raise ValueError("第一张图必须为 HxWx3 的 numpy 数组")

    h, w = first_image.shape[:2]
    bgr_flags = _normalize_assume_bgr(assume_bgr, len(images))
    prompt = _build_point_prompt(text_prompt, all_prompt=all_prompt)
    content: List[Dict[str, Any]] = []
    for image, flag in zip(images, bgr_flags):
        data_uri = _image_to_data_uri(image, assume_bgr=flag)
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    content.append({"type": "text", "text": prompt})

    messages: Sequence[Dict[str, Any]] = [{
        "role": "user",
        "content": content,
    }]

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    generated = resp.choices[0].message.content or ""
    raw = generated or None
    points = omni_decode_points(generated)
    if not points and not return_raw:
        raise RuntimeError(f"未解析到坐标，模型输出: {generated}")

    result = _scale_points_to_image(
        points,
        width=w,
        height=h,
        norm_range=norm_range,
    )
    if return_raw:
        return result, raw
    return result


# ---- 模型调用主入口 ----
def predict_point_from_rgb(
    image: np.ndarray,
    text_prompt: str,
    all_prompt: str = "",
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    assume_bgr: bool = True,
    return_raw: bool = False,
    norm_range: float = 1000.0,
    temperature: float = 0.7,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 512,
) -> Tuple[float, float] | Tuple[Optional[Tuple[float, float]], Optional[str]]:
    """
    发送单张图像到 vLLM/OpenAI 兼容接口，返回首个像素点坐标 (u, v)。

    参数:
        image: HxWx3 图像数组 (RGB 或 BGR)。
        all
        text_prompt: 描述目标的自然语言提示。
        base_url/model_name/api_key: vLLM/OpenAI 兼容服务配置。
        assume_bgr: True 表示输入为 BGR（OpenCV 常用）；False 表示已是 RGB。
        norm_range: 如果模型输出在 [0, norm_range]，会按图像宽高缩放到像素坐标。
    """
    points_result = predict_multi_points_from_multi_image(
        images=[image],
        text_prompt=text_prompt,
        all_prompt=all_prompt,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        assume_bgr=assume_bgr,
        norm_range=norm_range,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        return_raw=True,
    )
    points, generated = points_result
    if not points:
        if return_raw:
            return None, generated
        raise RuntimeError(f"未解析到坐标，模型输出: {generated}")

    pt = np.array(points[0], dtype=np.float64).reshape(-1)
    if return_raw:
        return (float(pt[0]), float(pt[1])), generated
    return float(pt[0]), float(pt[1])


def predict_multi_points_from_rgb(
    image: np.ndarray,
    text_prompt: str,
    all_prompt: str = "",
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    assume_bgr: bool = True,
    norm_range: float = 1000.0,
    temperature: float = 0.7,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 512,
    return_raw: bool = False,
) -> Union[List[Tuple[float, float]], Tuple[List[Tuple[float, float]], Optional[str]]]:
    """
    发送单张图像到 vLLM/OpenAI 兼容接口，返回像素点坐标列表。

    参数:
        image: HxWx3 图像数组 (RGB 或 BGR)。
        all
        text_prompt: 描述目标的自然语言提示。
        base_url/model_name/api_key: vLLM/OpenAI 兼容服务配置。
        assume_bgr: True 表示输入为 BGR（OpenCV 常用）；False 表示已是 RGB。
        norm_range: 如果模型输出在 [0, norm_range]，会按图像宽高缩放到像素坐标。
    """
    return predict_multi_points_from_multi_image(
        images=[image],
        text_prompt=text_prompt,
        all_prompt=all_prompt,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        assume_bgr=assume_bgr,
        norm_range=norm_range,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        return_raw=return_raw,
    )


__all__ = [
    "predict_point_from_rgb",
    "predict_multi_points_from_multi_image",
    "predict_multi_points_from_rgb",
    "omni_decode_points",
]
