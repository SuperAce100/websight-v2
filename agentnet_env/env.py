from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover - optional
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from .actions import Action
from .backend import PyAutoGuiBackend
from .config import EnvConfig
from .logging import TraceLogger
from .parser import parse_code_to_action
from .screen import ScreenManager


class AgentNetDesktopEnv:
    def __init__(
        self, config: Optional[EnvConfig] = None, task_id: Optional[str] = None
    ) -> None:
        self.config = config or EnvConfig()
        self.screen = ScreenManager(
            monitor_index=self.config.monitor_index, region=self.config.capture_region
        )
        self.backend = PyAutoGuiBackend(self.config, self.screen)
        self.logger = TraceLogger(self.config, self.screen, task_id=task_id)
        self._step_idx = 0

        self.observation_space = None
        if spaces is not None:
            # Note: actual image size may differ; this is a hint for RL
            h = self.screen.geometry.height
            w = self.screen.geometry.width
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, 3), dtype="uint8"
            )
        self.action_space = None

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        img = self.screen.grab()
        obs = img  # Return PIL image as observation for now
        self._step_idx = 0
        return obs, {"step": self._step_idx}

    def step(
        self, action: Action | Dict[str, Any] | str
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._step_idx += 1
        ts_start = self.screen.now_utc_iso()
        cap = self.logger.capture(self._step_idx)

        # Normalize action input
        raw_code: Optional[str] = None
        if isinstance(action, str):
            raw_code = action
            act = parse_code_to_action(action)
        elif isinstance(action, dict):
            # Assume already in internal schema
            act = Action.model_validate(action)  # type: ignore[attr-defined]
        else:
            act = action

        # Execute
        backend_info: Dict[str, Any] = {}
        error_msg: Optional[str] = None
        success = True
        try:
            backend_info = self.backend.execute(act)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            success = False
            error_msg = str(exc)
            self.logger.write_error(self._step_idx, exc)

        # Capture post
        cap = self.logger.finalize(cap, self._step_idx)
        ts_end = self.screen.now_utc_iso()

        # Log event
        self.logger.write_step(
            step=self._step_idx,
            ts_start=ts_start,
            ts_end=ts_end,
            raw_code=raw_code,
            action_payload=act.model_dump()
            if hasattr(act, "model_dump")
            else dict(act),  # type: ignore[arg-type]
            backend_info=backend_info,
            cap=cap,
            success=success,
            error=error_msg,
        )

        # Observation: latest screen image
        obs = cap.post_img if cap.post_img is not None else self.screen.grab()
        reward = 0.0
        # Done handling: if action is terminate, mark terminated
        terminated = getattr(act, "type", None) == "terminate"
        truncated = False
        info: Dict[str, Any] = {
            "success": success,
            "error": error_msg,
            "step": self._step_idx,
        }
        return obs, reward, terminated, truncated, info

    def observe(self):
        return self.screen.grab()

    def close(self) -> None:
        try:
            self.logger.close()
        except Exception:
            pass
