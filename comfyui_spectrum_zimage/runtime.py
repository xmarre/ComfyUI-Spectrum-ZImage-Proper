from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .config import SpectrumConfig
from .forecast import ChebyshevSpectrumForecaster


@dataclass(slots=True)
class RuntimeStats:
    actual_forward_count: int = 0
    forecasted_count: int = 0
    total_steps: int = 0
    current_window: float = 0.0
    run_id: int = 0
    forecast_disabled: bool = False
    disable_reason: Optional[str] = None
    sampler_name: Optional[str] = None


@dataclass(slots=True)
class _ActiveRun:
    run_id: int
    sampler_name: str
    total_steps: int
    schedule_values: tuple[float, ...]
    schedule_coords: tuple[float, ...]
    supports_solver_steps: bool
    next_solver_step_id: int = 0


@dataclass(slots=True)
class _ActiveSubcall:
    signature: tuple[Any, ...]
    expected_shape: Optional[tuple[int, ...]] = None
    observed_actual: bool = False
    used_forecast: bool = False
    predicted_feature: Optional[torch.Tensor] = None


@dataclass(slots=True)
class _ActiveStep:
    solver_step_id: int
    time_coord: float
    decision: Dict[str, Any]
    subcall_counts: Dict[tuple[Any, ...], int] = field(default_factory=dict)
    latest_subcall_key_by_signature: Dict[tuple[Any, ...], tuple[Any, ...]] = field(default_factory=dict)
    subcalls: Dict[tuple[Any, ...], _ActiveSubcall] = field(default_factory=dict)


class SpectrumRuntime:
    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg.validate()
        self.forecaster = ChebyshevSpectrumForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            max_history=self.cfg.max_history,
        )
        self.run_id = 0
        self.stats = RuntimeStats(current_window=float(self.cfg.window_size))
        self._active_run: Optional[_ActiveRun] = None
        self._forecasters: Dict[tuple[Any, ...], ChebyshevSpectrumForecaster] = {}
        self._active_steps: Dict[int, _ActiveStep] = {}
        self._reset_scheduler_state()

    @property
    def min_fit_points(self) -> int:
        return max(2, self.cfg.degree + 1)

    @property
    def active_run_id(self) -> Optional[int]:
        if self._active_run is None:
            return None
        return self._active_run.run_id

    @property
    def next_solver_step_id(self) -> int:
        if self._active_run is None:
            return 0
        return self._active_run.next_solver_step_id

    @property
    def active_run_supports_solver_steps(self) -> bool:
        if self._active_run is None:
            return False
        return bool(self._active_run.supports_solver_steps)

    def _reset_scheduler_state(self) -> None:
        self.curr_ws = float(self.cfg.window_size)
        self.num_consecutive_cached_steps = 0
        self.forecast_disabled = False
        self.forecast_disable_reason: Optional[str] = None
        self.forecaster.reset()
        self._forecasters = {}
        self._active_steps = {}
        self.stats.current_window = float(self.cfg.window_size)
        self.stats.forecast_disabled = False
        self.stats.disable_reason = None

    def reset_all(self) -> None:
        self.run_id += 1
        self.stats = RuntimeStats(current_window=float(self.cfg.window_size), run_id=self.run_id)
        self._active_run = None
        self._reset_scheduler_state()

    def _disable_forecasting(self, reason: str) -> None:
        if self.forecast_disabled:
            return
        self.forecast_disabled = True
        self.forecast_disable_reason = reason
        self.num_consecutive_cached_steps = 0
        self.curr_ws = float(self.cfg.window_size)
        self.forecaster.reset()
        self._forecasters = {}
        for step in self._active_steps.values():
            for subcall in step.subcalls.values():
                subcall.predicted_feature = None
        self.stats.current_window = self.curr_ws
        self.stats.forecast_disabled = True
        self.stats.disable_reason = reason

    def _normalize_branch_signature(self, branch_signature: Optional[tuple[Any, ...]]) -> tuple[Any, ...]:
        if not branch_signature:
            return (("__spectrum_default_branch__", True),)
        return tuple(branch_signature)

    def _get_or_create_branch_forecaster(self, signature: tuple[Any, ...]) -> ChebyshevSpectrumForecaster:
        forecaster = self._forecasters.get(signature)
        if forecaster is None:
            forecaster = ChebyshevSpectrumForecaster(
                degree=self.cfg.degree,
                ridge_lambda=self.cfg.ridge_lambda,
                max_history=self.cfg.max_history,
            )
            self._forecasters[signature] = forecaster
        else:
            forecaster.configure(
                degree=self.cfg.degree,
                ridge_lambda=self.cfg.ridge_lambda,
                max_history=self.cfg.max_history,
            )
        return forecaster

    def _get_branch_forecaster(self, signature: tuple[Any, ...]) -> Optional[ChebyshevSpectrumForecaster]:
        return self._forecasters.get(signature)

    def _branch_occurrence_key(self, signature: tuple[Any, ...], occurrence_index: int) -> tuple[Any, ...]:
        if occurrence_index <= 0:
            return signature
        return signature + (("__spectrum_occurrence__", occurrence_index),)

    def _require_subcall(
        self,
        step: _ActiveStep,
        branch_signature: Optional[tuple[Any, ...]],
    ) -> _ActiveSubcall:
        if branch_signature is None:
            default_signature = self._normalize_branch_signature(None)
            default_subcall_key = step.latest_subcall_key_by_signature.get(default_signature, default_signature)
            default_subcall = step.subcalls.get(default_subcall_key)
            if default_subcall is not None:
                return default_subcall
            if len(step.subcalls) == 1:
                return next(iter(step.subcalls.values()))
        signature = self._normalize_branch_signature(branch_signature)
        subcall_key = step.latest_subcall_key_by_signature.get(signature)
        if subcall_key is None:
            subcall_key = signature
        subcall = step.subcalls.get(subcall_key)
        if subcall is None:
            raise RuntimeError(f"Spectrum branch signature {signature!r} is not active for solver step {step.solver_step_id}.")
        return subcall

    def _any_branch_ready(self) -> bool:
        for forecaster in self._forecasters.values():
            if forecaster.ready(self.min_fit_points):
                return True
        return False

    def num_steps(self) -> int:
        if self.stats.total_steps > 0:
            return self.stats.total_steps
        return 50

    @staticmethod
    def _build_schedule_coords(sample_sigmas: torch.Tensor) -> tuple[tuple[float, ...], tuple[float, ...]]:
        values = tuple(float(v) for v in sample_sigmas.detach().flatten().tolist()[:-1])
        if not values:
            return (), ()

        start = values[0]
        end = values[-1]
        denom = end - start
        if abs(denom) < 1e-12:
            coords = tuple(0.0 for _ in values)
        else:
            coords = tuple(((v - start) / denom) * 2.0 - 1.0 for v in values)
        return values, coords

    def time_coord_for_step(self, solver_step_id: int) -> float:
        if self._active_run is None:
            raise RuntimeError("Spectrum runtime is not inside an active run.")
        idx = int(solver_step_id)
        if idx < 0 or idx >= len(self._active_run.schedule_coords):
            raise RuntimeError(f"Spectrum solver step {solver_step_id} is outside the active schedule.")
        return float(self._active_run.schedule_coords[idx])

    def _is_tail_actual_step(self, solver_step_id: int) -> bool:
        if self._active_run is None:
            return False
        tail_actual_steps = int(self.cfg.tail_actual_steps)
        if tail_actual_steps <= 0:
            return False
        tail_start = max(0, self._active_run.total_steps - tail_actual_steps)
        return int(solver_step_id) >= tail_start

    def start_run(self, sample_sigmas: torch.Tensor, sampler_name: str, *, supports_solver_steps: bool) -> int:
        self.run_id += 1
        schedule_values, schedule_coords = self._build_schedule_coords(sample_sigmas)
        total_steps = max(len(schedule_coords), 1)
        self.stats = RuntimeStats(
            current_window=float(self.cfg.window_size),
            total_steps=total_steps,
            run_id=self.run_id,
            sampler_name=sampler_name,
        )
        self._active_run = _ActiveRun(
            run_id=self.run_id,
            sampler_name=sampler_name,
            total_steps=total_steps,
            schedule_values=schedule_values,
            schedule_coords=schedule_coords,
            supports_solver_steps=bool(supports_solver_steps),
        )
        self._reset_scheduler_state()
        self.stats.run_id = self.run_id
        self.stats.total_steps = total_steps
        self.stats.sampler_name = sampler_name
        if not supports_solver_steps:
            self._disable_forecasting(
                f"sampler {sampler_name!r} does not expose one predict_noise call per solver step"
            )
        return self.run_id

    def end_run(self, run_id: int) -> None:
        if self._active_run is None or self._active_run.run_id != int(run_id):
            return
        self._active_run = None
        self._active_steps = {}

    def begin_solver_step(self, run_id: int, solver_step_id: int, time_coord: float, total_steps: int) -> Dict[str, Any]:
        if self._active_run is None or self._active_run.run_id != int(run_id):
            raise RuntimeError("Spectrum runtime is not inside the requested sampling run.")

        if int(total_steps) != self._active_run.total_steps:
            self._disable_forecasting("solver-step total_steps changed inside one sampling run")

        expected_time_coord = self.time_coord_for_step(int(solver_step_id))
        if not math.isclose(float(time_coord), expected_time_coord, rel_tol=0.0, abs_tol=1e-8):
            self._disable_forecasting("solver-step time_coord did not match the active schedule")

        existing = self._active_steps.get(int(solver_step_id))
        if existing is not None:
            return existing.decision

        if int(solver_step_id) != self._active_run.next_solver_step_id:
            self._disable_forecasting("solver-step ids are not sequential within the sampling run")

        actual_forward = True
        tail_actual_only = self._is_tail_actual_step(int(solver_step_id))
        if (
            not self.forecast_disabled
            and not tail_actual_only
            and int(solver_step_id) >= self.cfg.warmup_steps
            and self._any_branch_ready()
        ):
            ws_floor = max(1, math.floor(self.curr_ws))
            actual_forward = ((self.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if self.forecast_disabled or tail_actual_only or not self._any_branch_ready():
            actual_forward = True

        decision = {
            "run_id": int(run_id),
            "solver_step_id": int(solver_step_id),
            "time_coord": float(time_coord),
            "total_steps": int(total_steps),
            "actual_forward": bool(actual_forward),
            "forecast_disabled": self.forecast_disabled,
        }
        self._active_steps[int(solver_step_id)] = _ActiveStep(
            solver_step_id=int(solver_step_id),
            time_coord=float(time_coord),
            decision=decision,
        )
        self._active_run.next_solver_step_id = int(solver_step_id) + 1
        self.stats.current_window = self.curr_ws
        return decision

    def get_step_decision(self, run_id: int, solver_step_id: int) -> Optional[Dict[str, Any]]:
        step = self._active_steps.get(int(solver_step_id))
        if step is None or step.decision["run_id"] != int(run_id):
            return None
        return step.decision

    def step_used_forecast(self, run_id: int, solver_step_id: int) -> bool:
        return any(subcall.used_forecast for subcall in self._require_active_step(run_id, solver_step_id).subcalls.values())

    def register_model_hook_call(
        self,
        run_id: int,
        solver_step_id: int,
        *,
        expected_shape: tuple[int, ...],
        branch_signature: Optional[tuple[Any, ...]] = None,
    ) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        signature = self._normalize_branch_signature(branch_signature)
        occurrence_index = step.subcall_counts.get(signature, 0)
        step.subcall_counts[signature] = occurrence_index + 1
        subcall_key = self._branch_occurrence_key(signature, occurrence_index)
        step.latest_subcall_key_by_signature[signature] = subcall_key

        step.subcalls[subcall_key] = _ActiveSubcall(
            signature=subcall_key,
            expected_shape=tuple(expected_shape),
        )

    def observe_actual_feature(self, run_id: int, solver_step_id: int, feature: torch.Tensor) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        subcall = self._require_subcall(step, None)
        subcall.observed_actual = True
        subcall.used_forecast = False
        subcall.predicted_feature = None
        if self.forecast_disabled:
            return
        try:
            forecaster = self._get_or_create_branch_forecaster(subcall.signature)
            forecaster.update(step.time_coord, feature)
        except ValueError as exc:
            self._disable_forecasting(str(exc))

    def observe_actual_feature_for_branch(
        self,
        run_id: int,
        solver_step_id: int,
        feature: torch.Tensor,
        *,
        branch_signature: Optional[tuple[Any, ...]] = None,
    ) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        subcall = self._require_subcall(step, branch_signature)
        subcall.observed_actual = True
        subcall.used_forecast = False
        subcall.predicted_feature = None
        if self.forecast_disabled:
            return
        try:
            forecaster = self._get_or_create_branch_forecaster(subcall.signature)
            forecaster.update(step.time_coord, feature)
        except ValueError as exc:
            self._disable_forecasting(str(exc))

    def predict_feature(
        self,
        run_id: int,
        solver_step_id: int,
        *,
        expected_shape: Optional[tuple[int, ...]] = None,
        branch_signature: Optional[tuple[Any, ...]] = None,
    ) -> Optional[torch.Tensor]:
        step = self._require_active_step(run_id, solver_step_id)
        if step.decision["actual_forward"]:
            return None
        subcall = self._require_subcall(step, branch_signature)
        forecaster = self._get_branch_forecaster(subcall.signature)
        if self.forecast_disabled or forecaster is None or not forecaster.ready(self.min_fit_points):
            return None

        if subcall.predicted_feature is None:
            subcall.predicted_feature = forecaster.predict(
                time_coord=step.time_coord,
                blend_weight=self.cfg.blend_weight,
            )

        if expected_shape is not None and tuple(subcall.predicted_feature.shape) != tuple(expected_shape):
            self._disable_forecasting("predicted feature shape did not match the current solver-step input")
            return None

        subcall.used_forecast = True
        return subcall.predicted_feature

    def finalize_solver_step(self, run_id: int, solver_step_id: int, *, used_forecast: bool) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        any_actual = any(subcall.observed_actual for subcall in step.subcalls.values())
        any_forecast = any(subcall.used_forecast for subcall in step.subcalls.values())

        if len(step.subcalls) == 0:
            self._disable_forecasting("solver step finished without any model-hook subcalls")
        if step.decision["actual_forward"] and not any_actual:
            self._disable_forecasting("solver step requested an actual forward but no actual feature was observed")
        if not any_actual and not any_forecast:
            self._disable_forecasting("solver step finished without an actual feature or a forecasted feature")

        if any_forecast and not any_actual:
            self.num_consecutive_cached_steps += 1
            self.stats.forecasted_count += 1
            step.decision["actual_forward"] = False
        else:
            if (
                not self.forecast_disabled
                and step.decision["actual_forward"]
                and step.solver_step_id >= self.cfg.warmup_steps
            ):
                self.curr_ws = round(self.curr_ws + float(self.cfg.flex_window), 6)
            self.num_consecutive_cached_steps = 0
            self.stats.actual_forward_count += 1
            step.decision["actual_forward"] = True

        self.stats.current_window = self.curr_ws
        self._active_steps.pop(int(solver_step_id), None)

    def _require_active_step(self, run_id: int, solver_step_id: int) -> _ActiveStep:
        if self._active_run is None or self._active_run.run_id != int(run_id):
            raise RuntimeError("Spectrum runtime is not inside the requested sampling run.")
        step = self._active_steps.get(int(solver_step_id))
        if step is None:
            raise RuntimeError(f"Spectrum solver step {solver_step_id} is not active.")
        return step
