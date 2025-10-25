from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field


class Action(BaseModel):
    type: str


class MouseMove(Action):
    type: Literal["move"] = "move"
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class MouseClick(Action):
    type: Literal["click"] = "click"
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    button: Literal["left", "right", "middle"] = "left"
    clicks: int = 1


class MouseDoubleClick(Action):
    type: Literal["double_click"] = "double_click"
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    button: Literal["left", "right", "middle"] = "left"


class MouseRightClick(Action):
    type: Literal["right_click"] = "right_click"
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class MouseDrag(Action):
    type: Literal["drag"] = "drag"
    # Optional start; when omitted, start is current cursor position
    x0: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y0: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    duration: Optional[float] = None


class MouseScroll(Action):
    type: Literal["scroll"] = "scroll"
    clicks: int


class KeyPress(Action):
    type: Literal["press"] = "press"
    key: str


class KeyDown(Action):
    type: Literal["keyDown"] = "keyDown"
    key: str


class KeyUp(Action):
    type: Literal["keyUp"] = "keyUp"
    key: str


class Hotkey(Action):
    type: Literal["hotkey"] = "hotkey"
    keys: Tuple[str, ...]


class TextWrite(Action):
    type: Literal["write"] = "write"
    text: str
    interval: float = 0.0


class Sleep(Action):
    type: Literal["sleep"] = "sleep"
    seconds: float


class Terminate(Action):
    type: Literal["terminate"] = "terminate"


ActionLike = (
    MouseMove
    | MouseClick
    | MouseDoubleClick
    | MouseRightClick
    | MouseDrag
    | MouseScroll
    | KeyPress
    | KeyDown
    | KeyUp
    | Hotkey
    | TextWrite
    | Sleep
    | Terminate
)
