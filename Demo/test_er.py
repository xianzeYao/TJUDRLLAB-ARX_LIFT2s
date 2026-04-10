from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ERModelConfig:
    base_url: str = "http://172.28.102.11:22002/v1"
    model_name: str = "Embodied-R1.5-SFT-0128"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    top_p: float = 0.8
    seed: int = 3407
    max_tokens: int = 512
    assume_bgr: bool = False


@dataclass
class ERResult:
    image_paths: list[Path]
    points: list[tuple[float, float]]
    raw_response: Optional[str]
    output_path: Optional[Path]


DEFAULT_MODEL_CONFIG = ERModelConfig()


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV 未安装，无法读取/保存图像。请在运行环境中安装 cv2。"
        ) from exc
    return cv2


def _resolve_image_paths(image_files: Optional[Sequence[str | Path]]) -> list[Path]:
    if not image_files:
        raise ValueError("image_files must be non-empty")

    resolved: list[Path] = []
    for item in image_files:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        resolved.append(path)
    return resolved


def _load_images(paths: Sequence[Path]) -> list[Any]:
    cv2 = _import_cv2()
    images: list[Any] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        images.append(image)
    return images


def _draw_points(image: Any, points: Sequence[tuple[float, float]]) -> Any:
    cv2 = _import_cv2()
    vis = image.copy()
    for idx, (u, v) in enumerate(points, start=1):
        center = (int(round(u)), int(round(v)))
        cv2.circle(vis, center=center, radius=5,
                   color=(0, 0, 255), thickness=-1)
        cv2.putText(
            vis,
            text=str(idx),
            org=(center[0] - 4, center[1] + 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(0, 255, 0),
            thickness=1,
        )
    return vis


def _predict_points(
    images: Sequence[Any],
    *,
    prompt: str,
    all_prompt: Optional[str],
    config: ERModelConfig,
) -> tuple[list[tuple[float, float]], Optional[str]]:
    from utils import (
        predict_multi_points_from_multi_image,
        predict_multi_points_from_rgb,
    )

    effective_all_prompt = all_prompt or ""
    effective_text_prompt = prompt

    common_kwargs = {
        "text_prompt": effective_text_prompt,
        "all_prompt": effective_all_prompt,
        "base_url": config.base_url,
        "model_name": config.model_name,
        "api_key": config.api_key,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "return_raw": True,
    }
    if len(images) == 1:
        return predict_multi_points_from_rgb(
            image=images[0],
            assume_bgr=config.assume_bgr,
            **common_kwargs,
        )
    return predict_multi_points_from_multi_image(
        images=list(images),
        assume_bgr=[config.assume_bgr] * len(images),
        **common_kwargs,
    )


def test_er(
    *image_files: str | Path,
    prompt: str = "",
    all_prompt: Optional[str] = None,
    out_path: Optional[str | Path] = None,
    show: bool = False,
    save_vis: bool = False,
    model_config: ERModelConfig = DEFAULT_MODEL_CONFIG,
) -> ERResult:
    """
    纯函数式 ER 调用入口。

    传一个或多个图像路径，返回解析出的全部点。
    多图时会调用多图接口；返回值始终是点列表。
    """
    image_paths = _resolve_image_paths(list(image_files))
    images = _load_images(image_paths)
    points, raw = _predict_points(
        images,
        prompt=prompt,
        all_prompt=all_prompt,
        config=model_config,
    )

    saved_output_path: Optional[Path] = None
    if save_vis:
        if out_path is None:
            raise ValueError("out_path is required when save_vis=True")
        cv2 = _import_cv2()
        saved_output_path = Path(out_path).expanduser()
        if not saved_output_path.is_absolute():
            saved_output_path = (Path.cwd() / saved_output_path).resolve()
        saved_output_path.parent.mkdir(parents=True, exist_ok=True)
        vis = _draw_points(images[0], points)
        if not cv2.imwrite(str(saved_output_path), vis):
            raise RuntimeError(f"保存结果失败: {saved_output_path}")
        if show:
            cv2.imshow("Predicted Points", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return ERResult(
        image_paths=image_paths,
        points=points,
        raw_response=raw,
        output_path=saved_output_path,
    )


def main() -> None:
    image_paths = [
        "/path/to/img1.png",
        "/path/to/img2.png",
    ]
    if any(path.startswith("/path/to/") for path in image_paths):
        raise SystemExit(
            "Edit main() image_paths to real files before running.")

    result = test_er(
        *image_paths,
        prompt="",
        all_prompt="Point out all relevant points for this task.",
        # save_vis=True,
        # out_path="/path/to/out.png",
    )
    print(f"image_paths={result.image_paths}")
    print(f"points={result.points}")
    print(f"raw_response={result.raw_response}")


__all__ = [
    "ERModelConfig",
    "ERResult",
    "DEFAULT_MODEL_CONFIG",
    "test_er",
    "main",
]


if __name__ == "__main__":
    main()
