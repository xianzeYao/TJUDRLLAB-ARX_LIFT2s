import ast
import numpy as np
import cv2
import base64
import re
import os
import tempfile
import json
from typing import Any, Dict, List, Union, Tuple, Optional
from openai import OpenAI

def _preprocess_text(text: str) -> str:
    text = re.sub(r"```(?:json|python|html)?\n?(.*?)\n?```",
                  r"\1", text, flags=re.DOTALL)
    tag_match = re.search(r"<(?:point|points)>(.*?)</(?:point|points)>",
                          text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        text = tag_match.group(1)
    return text.strip()

# ---- 新增：通用 JSON 解析工具 ----
def _decode_json_result(text: str) -> Dict[str, Any]:
    """
    针对任务状态检测优化的解析函数。
    提取 status, has_error 和 description。
    """
    # 1. 基础清理
    text = _preprocess_text(text)
    
    # 2. 尝试提取 JSON 结构
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    json_str = match.group(1) if match else text
    
    # 尝试标准 JSON 和 Python 字面量解析
    parsed_data = None
    try:
        parsed_data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        try:
            data = ast.literal_eval(json_str)
            if isinstance(data, dict):
                parsed_data = data
        except:
            pass

    if isinstance(parsed_data, dict):
        # 统一 key 的格式，防止模型输出大写或略微偏差
        normalized_data = {k.lower(): v for k, v in parsed_data.items()}
        return {
            "status": normalized_data.get("status", "unknown"),
            "has_error": normalized_data.get("has_error", False),
            "description": normalized_data.get("description", text),
            "raw_parse_failed": False
        }

    # 3. 兜底策略：如果模型没按 JSON 输出，进行正则关键词匹配
    lower_text = text.lower()
    
    # 匹配状态关键词
    status = "unknown"
    if "success" in lower_text:
        status = "success"
    elif "failed" in lower_text or "error" in lower_text:
        status = "failed"
    elif "in_progress" in lower_text or "moving" in lower_text:
        status = "in_progress"
    elif "not_started" in lower_text:
        status = "not_started"

    return {
        "status": status,
        "has_error": status == "failed",
        "description": text,
        "raw_parse_failed": True
    }


def _get_video_b64(video_input: Union[str, cv2.VideoCapture], max_frames: int = 8, fps: int = 1) -> str:
    """
    核心处理：从路径或视频对象中采样 8 帧，并合成轻量级 MP4 Base64。
    """
    if isinstance(video_input, str):
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"视频文件不存在: {video_input}")
        cap = cv2.VideoCapture(video_input)
    else:
        cap = video_input

    # 1. 采样帧
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # 针对无法直接获取帧数的流式对象，尝试简单读取
        ret, frame = cap.read()
        if not ret: raise RuntimeError("无法读取视频内容")
        frames.append(cv2.resize(frame, (448, 448)))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # 缩放到 448x448 以保证传输效率且不失关键动作细节
                frames.append(cv2.resize(frame, (448, 448)))

    if isinstance(video_input, str):
        cap.release()

    if not frames:
        raise RuntimeError("未能从视频中提取到有效帧")

    # 2. 合成临时 MP4 视频
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        h, w, _ = frames[0].shape
        # 使用 avc1 (H.264) 编码，vLLM 兼容性最好
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        
        with open(tmp.name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

# ---- 修改后：基于视频的错误检测主函数 ----
def predict_video_error_check(
    video_input: Union[str, cv2.VideoCapture],
    text_prompt: str,
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    max_frames: int = 8,
    fps: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 512,
    return_raw: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], str]]:
    """
    通过 video_url 方式调用 vLLM 进行视频错误检测。
    """
    # 1. 获取视频 Base64
    video_b64 = _get_video_b64(video_input, max_frames=max_frames, fps=fps)

    # 2. 构建符合 vLLM 规范的 Content List
    content_list = [
        {
            "type": "video_url",
            "video_url": {
                "url": f"data:video/mp4;base64,{video_b64}",
                "fps": fps
            }
        },
        {
            "type": "text",
            "text": (
                f"Task to analyze: {text_prompt}.\n"
                "Based on the video sequence, categorize the task status into one of the following:\n"
                "1. 'not_started': The robot or actor has not yet begun moving toward the objective.\n"
                "2. 'in_progress': The action is currently happening but not yet completed.\n"
                "3. 'success': The task was successfully completed within the video clip.\n"
                "4. 'failed': An error occurred, or the task was attempted but failed (e.g., dropped the object, missed the target).\n\n"
                "Respond ONLY in valid JSON format with these keys:\n"
                "- \"status\": string (choose from: 'not_started', 'in_progress', 'success', 'failed')\n"
                "- \"has_error\": boolean (true if the status is 'failed', otherwise false)\n"
                "- \"description\": string (a brief explanation of the visual evidence)"
            )
        }
    ]

    # 3. API 调用
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content_list}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generated = resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"API 调用失败: {e}")

    # 4. 解析
    result = _decode_json_result(generated)
    return result