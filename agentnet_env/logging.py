from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageFilter

try:
    import imagehash  # type: ignore
except Exception:  # pragma: no cover - optional
    imagehash = None  # type: ignore

from .config import EnvConfig, LoggingLevel
from .screen import ScreenManager


def _fsync(f) -> None:
    f.flush()
    os.fsync(f.fileno())


@dataclass
class StepCapture:
    pre_img: Optional[Image.Image]
    post_img: Optional[Image.Image]
    pre_path: Optional[str]
    post_path: Optional[str]
    pre_hash: Optional[str]
    post_hash: Optional[str]


class TraceLogger:
    def __init__(
        self, config: EnvConfig, screen: ScreenManager, task_id: Optional[str] = None
    ) -> None:
        self.config = config
        self.screen = screen
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_name = f"{ts}_{task_id}" if task_id else ts
        self.run_dir = Path(config.log_dir) / run_name
        self.screens_dir = self.run_dir / "screens"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.screens_dir.mkdir(parents=True, exist_ok=True)

        self.actions_path = self.run_dir / "actions.jsonl"
        self.errors_path = self.run_dir / "errors.jsonl"
        self.meta_path = self.run_dir / "meta.json"

        # Initialize files
        with open(self.meta_path, "w", encoding="utf-8") as f:
            meta: Dict[str, Any] = {
                "config": self.config.model_dump(exclude_none=True),
                "monitor": {
                    "index": screen.geometry.monitor_index,
                    "width": screen.geometry.width,
                    "height": screen.geometry.height,
                    "left": screen.geometry.left,
                    "top": screen.geometry.top,
                },
                "versions": {},
            }
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._video_writer = None
        if self.config.save_video:
            try:
                import cv2  # type: ignore

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                size = (self.screen.geometry.width, self.screen.geometry.height)
                self._video_writer = cv2.VideoWriter(
                    str(self.run_dir / "video.mp4"), fourcc, 4.0, size
                )
            except Exception:
                self._video_writer = None

    def _hash(self, img: Image.Image) -> Optional[str]:
        if not self.config.enable_hashes or imagehash is None:
            return None
        try:
            return str(imagehash.phash(img))
        except Exception:
            return None

    def _apply_redactions(self, img: Image.Image) -> Image.Image:
        if not self.config.redaction_boxes:
            return img
        out = img.copy()
        w, h = out.size
        for box in self.config.redaction_boxes:
            x0 = int(box.x0 * w)
            y0 = int(box.y0 * h)
            x1 = int(box.x1 * w)
            y1 = int(box.y1 * h)
            region = out.crop((x0, y0, x1, y1)).filter(
                ImageFilter.GaussianBlur(radius=10)
            )
            out.paste(region, (x0, y0, x1, y1))
        return out

    def _save_image(self, img: Image.Image, step: int, kind: str) -> str:
        img = self._apply_redactions(img)
        path = self.screens_dir / f"{step:06d}_{kind}.png"
        img.save(path)
        return str(path.relative_to(self.run_dir))

    def capture(self, step: int) -> StepCapture:
        if self.config.logging_level == LoggingLevel.minimal:
            return StepCapture(None, None, None, None, None, None)
        pre = self.screen.grab()
        pre_path = (
            self._save_image(pre, step, "pre")
            if self.config.logging_level == LoggingLevel.verbose
            else None
        )
        pre_hash = self._hash(pre)
        return StepCapture(pre, None, pre_path, None, pre_hash, None)

    def finalize(self, cap: StepCapture, step: int) -> StepCapture:
        if self.config.logging_level == LoggingLevel.minimal:
            return cap
        post = self.screen.grab()
        post_path = self._save_image(post, step, "post")
        post_hash = self._hash(post)
        cap.post_img = post
        cap.post_path = post_path
        cap.post_hash = post_hash
        # Append to video if enabled
        if self._video_writer is not None:
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore

                frame = cv2.cvtColor(np.array(post), cv2.COLOR_RGB2BGR)
                self._video_writer.write(frame)
            except Exception:
                pass
        return cap

    def write_step(
        self,
        *,
        step: int,
        ts_start: str,
        ts_end: str,
        raw_code: Optional[str],
        action_payload: Dict[str, Any],
        backend_info: Dict[str, Any],
        cap: StepCapture,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        start_mono = time.monotonic()
        event: Dict[str, Any] = {
            "step": step,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "raw_code": raw_code,
            "action": action_payload,
            "latency_ms": int((time.monotonic() - start_mono) * 1000),
            "pre_image": cap.pre_path,
            "post_image": cap.post_path,
            "pre_hash": cap.pre_hash,
            "post_hash": cap.post_hash,
            "success": success,
            "error": error,
        }
        event.update(backend_info or {})
        with open(self.actions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            _fsync(f)

    def write_error(self, step: int, exc: Exception) -> None:
        tb = "".join(traceback.format_exception(exc))
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"step": step, "error": str(exc), "traceback": tb}) + "\n"
            )
            _fsync(f)

    def close(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
