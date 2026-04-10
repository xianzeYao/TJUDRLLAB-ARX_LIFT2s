from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import predict_point_from_rgb


COASTER_PROMPTS = [
    "the center of coaster (the left one near cups most)",
    "the center of coaster (the right one near cups most)",
    "the center of coaster (the leftmost one)",
    "the coaster (the rightmost one)",
]


def _build_single_prompts(
    step_cups: List[str],
    no_last_place: bool,
) -> List[str]:
    if not step_cups:
        raise ValueError("step_cups 不能为空")
    if len(step_cups) > len(COASTER_PROMPTS):
        raise ValueError(
            f"step_cups 数量不能超过 {len(COASTER_PROMPTS)}，当前为 {len(step_cups)}")

    prompts: List[str] = []
    for i, step in enumerate(step_cups):
        prompts.append(step)
        if not (no_last_place and i == len(step_cups) - 1):
            prompts.append(COASTER_PROMPTS[i])
    return prompts


def _predict_points(
    image_bgr: np.ndarray,
    prompts: List[str],
    temperature: float = 0.0,
) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for prompt in prompts:
        u, v = predict_point_from_rgb(
            image_bgr,
            text_prompt=prompt,
            temperature=temperature,
            assume_bgr=False,
        )
        pts.append((int(round(u)), int(round(v))))
    return pts


def _split_pick_place_points(
    pts: List[Tuple[int, int]],
    step_n: int,
    no_last_place: bool,
) -> Tuple[List[Tuple[int, int]], List[Optional[Tuple[int, int]]]]:
    needed_points = step_n * 2 - (1 if no_last_place else 0)
    if len(pts) < needed_points:
        raise RuntimeError(f"点数不足: {len(pts)}/{needed_points}")

    # 与 dual_arm_pick_planning_parallel 保持一致：微调第一个 pick 点
    pts = list(pts)
    pts[0] = (pts[0][0], pts[0][1] - 15)

    pick_px: List[Tuple[int, int]] = []
    place_px: List[Optional[Tuple[int, int]]] = []
    p_idx = 0
    for i in range(step_n):
        pick_px.append(pts[p_idx])
        p_idx += 1
        if not (no_last_place and i == step_n - 1):
            place_px.append(pts[p_idx])
            p_idx += 1
        else:
            place_px.append(None)
    return pick_px, place_px


def _draw_points(
    image_bgr: np.ndarray,
    pick_px: List[Tuple[int, int]],
    place_px: List[Optional[Tuple[int, int]]],
) -> np.ndarray:
    vis = image_bgr.copy()
    for i, p in enumerate(pick_px):
        cv2.circle(vis, p, 3, (0, 0, 255), -1)
        cv2.putText(vis, f"P{i+1}", (p[0] + 6, p[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for i, p in enumerate(place_px):
        if p is None:
            continue
        cv2.circle(vis, p, 3, (255, 0, 0), -1)
        cv2.putText(vis, f"C{i+1}", (p[0] + 6, p[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return vis


def vis4point(
    step_cups: List[str],
    image_path: str,
    no_last_place: bool = True,
    out_path: Optional[str] = None,
    temperature: float = 0.0,
) -> np.ndarray:
    """
    输入:
        step_cups: 步骤杯子列表，例如 ["green cup", "blue cup", "red cup"]。
        image_path: 输入图像路径。
    输出:
        标注后的 BGR 图像（包含点和 P/C 标注，不叠加左上角 prompt 文本）。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    prompts = _build_single_prompts(step_cups, no_last_place=no_last_place)
    pts = _predict_points(image, prompts, temperature=temperature)
    pick_px, place_px = _split_pick_place_points(
        pts, step_n=len(step_cups), no_last_place=no_last_place)
    vis = _draw_points(image, pick_px, place_px)

    if out_path is None:
        src = Path(image_path)
        out_path = str(src.with_name(f"{src.stem}_vis4point.png"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), vis)
    if not ok:
        raise RuntimeError(f"保存结果失败: {out}")
    print(f"vis4point 输出: {out}")
    return vis


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据步骤杯子列表对单张图像执行多次 pointout，并输出可视化图。")
    parser.add_argument(
        "--image-path",
        default="../Testdata4Mani/point.png",
        help="输入图像路径",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        required=True,
        help='步骤杯子列表，例如: --steps "green cup" "blue cup" "red cup"',
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="输出图像路径（默认在输入图同目录下生成 *_vis4point.png）",
    )
    parser.add_argument(
        "--last-place",
        action="store_true",
        help="默认不执行最后一步 place；设置该选项后执行最后一步 place。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="pointout 温度参数，默认 0.0",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    vis4point(
        step_cups=args.steps,
        image_path=args.image_path,
        no_last_place=not args.last_place,
        out_path=args.out_path,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
