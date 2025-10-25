from __future__ import annotations

import ast
from typing import Any, Dict, List

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


def _get_first(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


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

    if module not in (None, "pyautogui", "computer"):
        raise UnsupportedActionError(f"Unsupported module: {module}")

    # Convert args/kwargs to Python values
    kwargs: Dict[str, Any] = {}
    args: List[Any] = []
    for a in call.args:
        args.append(ast.literal_eval(a))
    for kw in call.keywords:
        kwargs[kw.arg] = ast.literal_eval(kw.value)

    # Map to internal actions with flexible naming (pyautogui.* or computer.*)
    n = name.lower()

    if n in {"moveto", "move_to", "move", "mouse_move", "move_mouse"}:
        x = _get_first(
            kwargs, ["x", "x_ratio", "xnorm", "xn"], args[0] if len(args) >= 1 else None
        )
        y = _get_first(
            kwargs, ["y", "y_ratio", "ynorm", "yn"], args[1] if len(args) >= 2 else None
        )
        return MouseMove(x=float(x), y=float(y))

    if n in {"click", "leftclick", "left_click"}:
        x = _get_first(
            kwargs, ["x", "x_ratio", "xnorm", "xn"], args[0] if len(args) >= 1 else None
        )
        y = _get_first(
            kwargs, ["y", "y_ratio", "ynorm", "yn"], args[1] if len(args) >= 2 else None
        )
        button = _kw(kwargs, "button", "left")
        clicks = int(_kw(kwargs, "clicks", 1))
        return MouseClick(x=float(x), y=float(y), button=button, clicks=clicks)

    if n in {"doubleclick", "double_click"}:
        x = _get_first(
            kwargs, ["x", "x_ratio", "xnorm", "xn"], args[0] if len(args) >= 1 else None
        )
        y = _get_first(
            kwargs, ["y", "y_ratio", "ynorm", "yn"], args[1] if len(args) >= 2 else None
        )
        button = _kw(kwargs, "button", "left")
        return MouseDoubleClick(x=float(x), y=float(y), button=button)

    if n in {"rightclick", "right_click"}:
        x = _get_first(
            kwargs, ["x", "x_ratio", "xnorm", "xn"], args[0] if len(args) >= 1 else None
        )
        y = _get_first(
            kwargs, ["y", "y_ratio", "ynorm", "yn"], args[1] if len(args) >= 2 else None
        )
        return MouseRightClick(x=float(x), y=float(y))

    if n in {"dragto", "drag_to", "drag"}:
        x0 = _get_first(kwargs, ["x0", "start_x", "x_start"], None)
        y0 = _get_first(kwargs, ["y0", "start_y", "y_start"], None)
        x1 = _get_first(
            kwargs, ["x1", "end_x", "x_end", "x"], args[0] if len(args) >= 1 else None
        )
        y1 = _get_first(
            kwargs, ["y1", "end_y", "y_end", "y"], args[1] if len(args) >= 2 else None
        )
        duration = float(_get_first(kwargs, ["duration", "secs", "seconds"], 0.2))
        return MouseDrag(
            x0=float(x0) if x0 is not None else None,
            y0=float(y0) if y0 is not None else None,
            x1=float(x1),
            y1=float(y1),
            duration=duration,
        )

    if n == "scroll":
        clicks = int(
            _get_first(
                kwargs, ["clicks", "amount", "delta"], args[0] if len(args) >= 1 else 0
            )
        )
        return MouseScroll(clicks=clicks)

    if n in {"press", "key", "key_press"}:
        key = _get_first(kwargs, ["key"], args[0] if len(args) >= 1 else None)
        return KeyPress(key=str(key))

    if n in {"keydown", "key_down"}:
        key = _get_first(kwargs, ["key"], args[0] if len(args) >= 1 else None)
        return KeyDown(key=str(key))

    if n in {"keyup", "key_up"}:
        key = _get_first(kwargs, ["key"], args[0] if len(args) >= 1 else None)
        return KeyUp(key=str(key))

    if n == "hotkey":
        keys = kwargs.get("keys")
        if keys is None:
            keys = tuple(args)
        return Hotkey(keys=tuple(str(k) for k in keys))

    if n in {"write", "type", "input", "text"}:
        text = _get_first(
            kwargs, ["text", "s", "value"], args[0] if len(args) >= 1 else ""
        )
        interval = float(
            _get_first(kwargs, ["interval"], args[1] if len(args) >= 2 else 0.0)
        )
        return TextWrite(text=str(text), interval=interval)

    if n in {"sleep", "wait", "delay"}:
        seconds = float(
            _get_first(
                kwargs,
                ["seconds", "secs", "duration"],
                args[0] if len(args) >= 1 else 0.0,
            )
        )
        return Sleep(seconds=seconds)

    if n in {"terminate", "end", "finish"}:
        from .actions import Terminate

        return Terminate()

    raise UnsupportedActionError(f"Unsupported action name: {name}")
