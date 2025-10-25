from __future__ import annotations

import time
from typing import Any, Dict, Tuple

try:
    import pyautogui
except Exception:  # pragma: no cover - optional runtime dep
    pyautogui = None  # type: ignore[assignment]

from .actions import (
    Action,
    Hotkey,
    KeyDown,
    KeyPress,
    KeyUp,
    MouseClick,
    MouseDoubleClick,
    MouseDrag,
    MouseMove,
    MouseRightClick,
    MouseScroll,
    Sleep,
    Terminate,
    TextWrite,
)
from .config import EnvConfig
from .screen import ScreenManager


class PyAutoGuiBackend:
    def __init__(self, config: EnvConfig, screen: ScreenManager) -> None:
        if pyautogui is None and not config.dry_run:
            raise RuntimeError("pyautogui is required when not in dry_run mode")

        self.config = config
        self.screen = screen

        if pyautogui is not None:
            pyautogui.PAUSE = max(0.0, float(config.pause_seconds))
            # Fail-safe allows moving cursor to a corner to abort; keep default True

    def _cursor_pos(self) -> Tuple[int, int]:
        if pyautogui is None:
            return (0, 0)
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    def _maybe_sleep(self, seconds: float | None) -> None:
        delay = self.config.action_delay_seconds + (seconds or 0.0)
        if delay > 0:
            time.sleep(delay)

    def execute(self, action: Action) -> Dict[str, Any]:
        cursor_before = self._cursor_pos()
        abs_coords: Tuple[int, int] | None = None

        if isinstance(action, MouseMove):
            abs_coords = self.screen.normalized_to_absolute(action.x, action.y)
            if pyautogui and not self.config.dry_run:
                pyautogui.moveTo(abs_coords[0], abs_coords[1])
            self._maybe_sleep(None)

        elif isinstance(action, MouseClick):
            abs_coords = self.screen.normalized_to_absolute(action.x, action.y)
            if pyautogui and not self.config.dry_run:
                pyautogui.click(
                    x=abs_coords[0],
                    y=abs_coords[1],
                    clicks=action.clicks,
                    button=action.button,
                )
            self._maybe_sleep(None)

        elif isinstance(action, MouseDoubleClick):
            abs_coords = self.screen.normalized_to_absolute(action.x, action.y)
            if pyautogui and not self.config.dry_run:
                pyautogui.doubleClick(
                    x=abs_coords[0], y=abs_coords[1], button=action.button
                )
            self._maybe_sleep(None)

        elif isinstance(action, MouseRightClick):
            abs_coords = self.screen.normalized_to_absolute(action.x, action.y)
            if pyautogui and not self.config.dry_run:
                pyautogui.rightClick(x=abs_coords[0], y=abs_coords[1])
            self._maybe_sleep(None)

        elif isinstance(action, MouseDrag):
            start = None
            if action.x0 is not None and action.y0 is not None:
                start = self.screen.normalized_to_absolute(action.x0, action.y0)
            end = self.screen.normalized_to_absolute(action.x1, action.y1)
            abs_coords = end
            if pyautogui and not self.config.dry_run:
                if start is not None:
                    pyautogui.moveTo(*start)
                pyautogui.dragTo(
                    end[0],
                    end[1],
                    duration=action.duration
                    if action.duration is not None
                    else self.config.drag_default_duration,
                )
            self._maybe_sleep(action.duration)

        elif isinstance(action, MouseScroll):
            if pyautogui and not self.config.dry_run:
                pyautogui.scroll(action.clicks)
            self._maybe_sleep(None)

        elif isinstance(action, KeyPress):
            if pyautogui and not self.config.dry_run:
                pyautogui.press(action.key)
            self._maybe_sleep(None)

        elif isinstance(action, KeyDown):
            if pyautogui and not self.config.dry_run:
                pyautogui.keyDown(action.key)
            self._maybe_sleep(None)

        elif isinstance(action, KeyUp):
            if pyautogui and not self.config.dry_run:
                pyautogui.keyUp(action.key)
            self._maybe_sleep(None)

        elif isinstance(action, Hotkey):
            if pyautogui and not self.config.dry_run:
                pyautogui.hotkey(*action.keys)
            self._maybe_sleep(None)

        elif isinstance(action, TextWrite):
            if pyautogui and not self.config.dry_run:
                pyautogui.write(action.text, interval=max(0.0, action.interval))
            self._maybe_sleep(None)

        elif isinstance(action, Sleep):
            time.sleep(max(0.0, action.seconds))

        elif isinstance(action, Terminate):
            # No-op at backend level; env may choose to end episode
            pass

        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported action type: {action.type}")

        cursor_after = self._cursor_pos()
        return {
            "cursor_before": {"x": cursor_before[0], "y": cursor_before[1]},
            "cursor_after": {"x": cursor_after[0], "y": cursor_after[1]},
            "abs_coords": {"x": abs_coords[0], "y": abs_coords[1]}
            if abs_coords
            else None,
        }
