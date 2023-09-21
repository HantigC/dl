from typing import Sequence, Tuple, Union, Optional
import cv2
import numpy as np
from .box import make_grid, xywh_to_yxyx
import itertools as it


def draw_grid_(
    img: np.ndarray,
    grid: Sequence[Sequence[int]] = None,
    grid_size: Tuple[int, int] = None,
) -> None:
    if grid is None:
        if grid_size is None:
            raise ValueError(
                "At least one of `grid_size` and `grid` should be provided"
            )

        grid = make_grid(grid_size)

    h, w, _ = img.shape
    for tl_y, tl_x, br_y, br_x in grid:
        cv2.rectangle(
            img,
            (int(w * tl_x), int(h * tl_y)),
            (int(w * br_x), int(h * br_y)),
            (255, 0, 0),
        )


def draw_rect_xywh_(
    img: np.ndarray,
    tl_x: Union[int, float],
    tl_y: Union[int, float],
    w: Union[int, float],
    h: Union[int, float],
    color: Tuple[int, int, int],
    **kwargs,
) -> None:
    tl_x = int(tl_x)
    tl_y = int(tl_y)
    w = int(w)
    h = int(h)
    cv2.rectangle(img, (tl_x, tl_y), (tl_x + w, tl_y + h), color=color, **kwargs)


def draw_bbox_xyhw(
    img: np.ndarray,
    category: str,
    tl_x: Union[int, float],
    tl_y: Union[int, float],
    w: Union[int, float],
    h: Union[int, float],
    box_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    **kwargs,
) -> None:
    draw_rect_xywh_(img, tl_x, tl_y, w, h, box_color, **kwargs)
    cv2.putText(img, category, (int(tl_x), int(tl_y)), 1, 1, text_color)


def draw_boxes_tlbr_(
    img,
    boxes: Sequence[Sequence[Union[int, float]]],
    colors: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    scale: bool = True,
    **kwargs,
) -> None:
    if colors is None:
        colors = np.random.randint(255, size=(len(boxes), 3)).tolist()
    elif isinstance(colors, list):
        assert len(boxes) == len(
            colors
        ), "`boxes` and `colors` should hange the same length"
        if not isinstance(colors[0], (list, tuple)):
            colors = it.cycle([colors])
    else:
        raise ValueError(
            f"`colors` should be a list of colors or a single color, not {colors}"
        )

    h, w, _ = img.shape
    for c, (tl_y, tl_x, br_y, br_x) in zip(colors, boxes):
        if scale:
            cv2.rectangle(
                img,
                (int(w * tl_x), int(h * tl_y)),
                (int(w * br_x), int(h * br_y)),
                color=c,
            )
        else:
            cv2.rectangle(
                img,
                (int(tl_x), int(tl_y)),
                (int(br_x), int(br_y)),
                color=c,
            )


def draw_boxes_xywh_(
    img,
    boxes: Sequence[Sequence[Union[int, float]]],
    colors: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    scale: bool = True,
    **kwargs,
) -> None:
    xywh = xywh_to_yxyx(boxes)
    draw_boxes_tlbr_(img, xywh, colors=colors, scale=scale, **kwargs)
