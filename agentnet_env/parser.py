from __future__ import annotations

import ast
from typing import Any, Dict

from .actions import (
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
    TextWrite,
)


class UnsupportedActionError(ValueError):
    pass


def _kw(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def parse_code_to_action(code: str):
    node = ast.parse(code.strip())
    if (
        not node.body
        or not isinstance(node.body[0], ast.Expr)
        or not isinstance(node.body[0].value, ast.Call)
    ):
        raise UnsupportedActionError("Expected a single function call expression")

    call = node.body[0].value
    # Support dotted names like pyautogui.click
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module = func.value.id
        name = func.attr
    elif isinstance(func, ast.Name):
        module = None
        name = func.id
    else:
        raise UnsupportedActionError("Unsupported call target")

    if module not in (None, "pyautogui"):
        raise UnsupportedActionError(f"Unsupported module: {module}")

    # Convert args/kwargs to Python values
    kwargs: Dict[str, Any] = {}
    for kw in call.keywords:
        kwargs[kw.arg] = ast.literal_eval(kw.value)

    # Map to internal actions
    if name == "moveTo" or name == "move":
        return MouseMove(x=float(kwargs["x"]), y=float(kwargs["y"]))
    if name == "click":
        return MouseClick(
            x=float(kwargs["x"]),
            y=float(kwargs["y"]),
            button=_kw(kwargs, "button", "left"),
            clicks=int(_kw(kwargs, "clicks", 1)),
        )
    if name == "doubleClick":
        return MouseDoubleClick(
            x=float(kwargs["x"]),
            y=float(kwargs["y"]),
            button=_kw(kwargs, "button", "left"),
        )
    if name == "rightClick":
        return MouseRightClick(x=float(kwargs["x"]), y=float(kwargs["y"]))
    if name == "dragTo":
        # Accept both forms: with x0,y0 provided, or implicit current position
        x0 = kwargs.get("x0")
        y0 = kwargs.get("y0")
        return MouseDrag(
            x0=float(x0) if x0 is not None else None,
            y0=float(y0) if y0 is not None else None,
            x1=float(kwargs["x1"]),
            y1=float(kwargs["y1"]),
            duration=float(_kw(kwargs, "duration", 0.2)),
        )
    if name == "scroll":
        return MouseScroll(clicks=int(kwargs["clicks"]))
    if name == "press":
        return KeyPress(key=str(kwargs["key"]))
    if name == "keyDown":
        return KeyDown(key=str(kwargs["key"]))
    if name == "keyUp":
        return KeyUp(key=str(kwargs["key"]))
    if name == "hotkey":
        return Hotkey(keys=tuple(kwargs["keys"]))
    if name == "write":
        return TextWrite(
            text=str(kwargs["text"]), interval=float(_kw(kwargs, "interval", 0.0))
        )
    if name == "sleep":
        return Sleep(seconds=float(kwargs["seconds"]))

    raise UnsupportedActionError(f"Unsupported action name: {name}")
