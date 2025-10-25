from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from mss import mss
from PIL import Image


@dataclass
class ScreenGeometry:
    width: int
    height: int
    left: int
    top: int
    monitor_index: int


class ScreenManager:
    def __init__(
        self, monitor_index: int = 0, region: Optional[Tuple[int, int, int, int]] = None
    ) -> None:
        self._sct = mss()
        self.monitor_index = monitor_index
        self.region = region

        # Determine geometry
        monitors = self._sct.monitors
        if monitor_index < 0 or monitor_index >= len(monitors):
            raise ValueError(
                f"Invalid monitor_index={monitor_index}; available monitors: {len(monitors) - 1}"
            )

        mon = monitors[monitor_index if monitor_index > 0 else 1]
        left = mon.get("left", 0)
        top = mon.get("top", 0)
        width = mon.get("width", 0)
        height = mon.get("height", 0)

        if region is not None:
            r_left, r_top, r_width, r_height = region
            left += r_left
            top += r_top
            width = r_width
            height = r_height

        self.geometry = ScreenGeometry(
            width=width, height=height, left=left, top=top, monitor_index=monitor_index
        )

    def normalized_to_absolute(self, x_norm: float, y_norm: float) -> Tuple[int, int]:
        x_abs = int(round(self.geometry.left + x_norm * self.geometry.width))
        y_abs = int(round(self.geometry.top + y_norm * self.geometry.height))
        return x_abs, y_abs

    def grab(self) -> Image.Image:
        bbox = {
            "left": self.geometry.left,
            "top": self.geometry.top,
            "width": self.geometry.width,
            "height": self.geometry.height,
        }
        img = self._sct.grab(bbox)
        # Convert to PIL
        arr = np.asarray(img)
        # MSS returns BGRA
        rgb = arr[..., :3][:, :, ::-1]
        return Image.fromarray(rgb)

    def grab_resized(self, size: Optional[Tuple[int, int]]) -> Image.Image:
        img = self.grab()
        if size is None:
            return img
        return img.resize(size, Image.Resampling.BILINEAR)

    def now_utc_iso(self) -> str:
        return (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            + f".{int((time.time() % 1) * 1000):03d}Z"
        )
