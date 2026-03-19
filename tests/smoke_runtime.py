from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum_zimage.config import SpectrumConfig
from comfyui_spectrum_zimage.forecast import ChebyshevSpectrumForecaster
from comfyui_spectrum_zimage.runtime import SpectrumRuntime
from comfyui_spectrum_zimage.zimage import (
    _SUPPORTED_SINGLE_EVAL_SAMPLERS,
    _forecast_feature_sanitization_stats,
    _sanitize_forecast_feature_for_final_layer,
    locate_zimage_inner_model,
)


def make_runtime(**overrides) -> SpectrumRuntime:
    cfg_kwargs = {
        "blend_weight": 0.5,
        "degree": 4,
        "ridge_lambda": 0.1,
        "window_size": 2.0,
        "flex_window": 0.75,
        "warmup_steps": 5,
        "tail_actual_steps": 3,
        "max_history": 128,
    }
    cfg_kwargs.update(overrides)
    cfg = SpectrumConfig(**cfg_kwargs).validate()
    return SpectrumRuntime(cfg)


def test_solver_step_scheduler() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        assert decision["actual_forward"] is True
        runtime.register_model_hook_call(
            run_id,
            step_id,
            expected_shape=(1, 8, 4),
            branch_signature=(("cond_or_uncond", (0, 1)),),
        )
        runtime.observe_actual_feature(decision["run_id"], decision["solver_step_id"], torch.randn(1, 8, 4))
        runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    runtime.register_model_hook_call(
        run_id,
        5,
        expected_shape=(1, 8, 4),
        branch_signature=(("cond_or_uncond", (0, 1)),),
    )
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4))
    assert predicted is not None
    assert predicted.shape == (1, 8, 4)
    runtime.finalize_solver_step(run_id, 5, used_forecast=not decision["actual_forward"])
    runtime.end_run(run_id)


def test_forecast_fallback_reconciles_bookkeeping() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    assert decision["actual_forward"] is False
    before_window = runtime.curr_ws

    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4))
    assert predicted is None
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(run_id, 5, used_forecast=False)

    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "predicted feature shape did not match the current solver-step input"
    assert runtime.stats.forecasted_count == 0
    assert runtime.stats.actual_forward_count == 6
    assert runtime.num_consecutive_cached_steps == 0
    assert runtime.curr_ws == before_window
    runtime.end_run(run_id)


def test_observe_actual_feature_clears_forecast_latch() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
        runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4))
    assert predicted is not None
    assert runtime.step_used_forecast(run_id, 5) is True
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4))
    assert runtime.step_used_forecast(run_id, 5) is False
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_unsupported_sampler_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_heun", supports_solver_steps=False)
    decision = runtime.begin_solver_step(
        run_id,
        0,
        runtime.time_coord_for_step(0),
        len(sample_sigmas) - 1,
    )
    assert decision["actual_forward"] is True
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "sampler 'sample_heun' does not expose one predict_noise call per solver step"
    runtime.register_model_hook_call(run_id, 0, expected_shape=(1, 8, 4))
    runtime.observe_actual_feature(run_id, 0, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(run_id, 0, used_forecast=False)
    runtime.end_run(run_id)


def test_inconsistent_hook_shape_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4))
    assert predicted is None
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "predicted feature shape did not match the current solver-step input"
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_cross_step_feature_shape_mismatch_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    first = runtime.begin_solver_step(
        run_id,
        0,
        runtime.time_coord_for_step(0),
        total_steps,
    )
    runtime.register_model_hook_call(run_id, 0, expected_shape=(1, 8, 4))
    runtime.observe_actual_feature(run_id, 0, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(first["run_id"], first["solver_step_id"], used_forecast=False)

    second = runtime.begin_solver_step(
        run_id,
        1,
        runtime.time_coord_for_step(1),
        total_steps,
    )
    runtime.register_model_hook_call(run_id, 1, expected_shape=(1, 16, 4))
    runtime.observe_actual_feature(run_id, 1, torch.randn(1, 16, 4))

    assert runtime.stats.forecast_disabled is True
    assert (
        runtime.stats.disable_reason
        == "Spectrum feature shape changed from (1, 8, 4) to (1, 16, 4)."
    )
    runtime.finalize_solver_step(second["run_id"], second["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_distinct_branch_signatures_within_one_solver_step_do_not_disable_forecast() -> None:
    runtime = make_runtime(degree=1, warmup_steps=2)
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    branch_a = (("cond_or_uncond", (0,)), ("uuids", ("cond-a",)))
    branch_b = (("cond_or_uncond", (1,)), ("uuids", ("uncond-b",)))

    for step_id in range(2):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        assert decision["actual_forward"] is True

        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 1), branch_signature=branch_a)
        runtime.observe_actual_feature_for_branch(
            run_id, step_id, torch.tensor([[1.0 + step_id]], dtype=torch.float32), branch_signature=branch_a
        )

        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 1), branch_signature=branch_b)
        runtime.observe_actual_feature_for_branch(
            run_id, step_id, torch.tensor([[10.0 + step_id]], dtype=torch.float32), branch_signature=branch_b
        )

        runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        2,
        runtime.time_coord_for_step(2),
        total_steps,
    )
    assert decision["actual_forward"] is False

    runtime.register_model_hook_call(run_id, 2, expected_shape=(1, 1), branch_signature=branch_a)
    assert runtime.predict_feature(run_id, 2, expected_shape=(1, 1), branch_signature=branch_a) is not None

    runtime.register_model_hook_call(run_id, 2, expected_shape=(1, 1), branch_signature=branch_b)
    assert runtime.predict_feature(run_id, 2, expected_shape=(1, 1), branch_signature=branch_b) is not None

    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=True)
    assert runtime.stats.forecast_disabled is False
    assert runtime.stats.forecasted_count == 1
    assert runtime.stats.actual_forward_count == 2
    runtime.end_run(run_id)


def test_repeated_same_branch_signature_within_one_solver_step_do_not_disable_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1
    branch_signature = (("cond_or_uncond", (0, 1)),)

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4), branch_signature=branch_signature)
        runtime.observe_actual_feature_for_branch(
            run_id,
            step_id,
            torch.randn(1, 8, 4),
            branch_signature=branch_signature,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4), branch_signature=branch_signature)
        runtime.observe_actual_feature_for_branch(
            run_id,
            step_id,
            torch.randn(1, 8, 4),
            branch_signature=branch_signature,
        )
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    assert decision["actual_forward"] is False

    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4), branch_signature=branch_signature)
    pred_a = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), branch_signature=branch_signature)
    assert pred_a is not None

    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4), branch_signature=branch_signature)
    pred_b = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), branch_signature=branch_signature)
    assert pred_b is not None

    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=True)
    assert runtime.stats.forecast_disabled is False
    assert runtime.stats.forecasted_count == 1
    runtime.end_run(run_id)


def test_nonuniform_schedule_coords_are_used() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.tensor([10.0, 9.0, 1.0, 0.0])
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)

    coords = [runtime.time_coord_for_step(i) for i in range(3)]
    assert math.isclose(coords[0], -1.0)
    assert math.isclose(coords[2], 1.0)
    assert math.isclose(coords[1], -7.0 / 9.0)
    assert not math.isclose(coords[1] - coords[0], coords[2] - coords[1])

    runtime.end_run(run_id)


def test_forecaster_respects_nonuniform_coords() -> None:
    forecaster = ChebyshevSpectrumForecaster(degree=1, ridge_lambda=0.1, max_history=16)
    forecaster.update(-1.0, torch.tensor([10.0]))
    forecaster.update(-7.0 / 9.0, torch.tensor([9.0]))

    pred = forecaster.predict(time_coord=1.0, blend_weight=0.0)
    assert torch.allclose(pred, torch.tensor([1.0]), atol=1e-5)


def test_locator_accepts_current_zimage_shape() -> None:
    class DummyLinear:
        in_features = 256

    class DummyFinal:
        adaLN_modulation = [object(), DummyLinear()]

    class DummyOwner:
        class ModelConfig:
            unet_config = {"z_image_modulation": True}
        model_config = ModelConfig()

    class DummyInner:
        def patchify_and_embed(self):
            pass
        def unpatchify(self):
            pass
        layers = []
        final_layer = DummyFinal()
        t_embedder = object()

    owner = DummyOwner()
    owner.diffusion_model = DummyInner()

    class DummyPatcher:
        model = owner

    inner, path = locate_zimage_inner_model(DummyPatcher())
    assert inner is owner.diffusion_model
    assert path == "model.diffusion_model"


def test_locator_rejects_non_zimage_lumina() -> None:
    class DummyLinear:
        in_features = 1024

    class DummyFinal:
        adaLN_modulation = [object(), DummyLinear()]

    class DummyInner:
        def patchify_and_embed(self):
            pass
        def unpatchify(self):
            pass
        layers = []
        final_layer = DummyFinal()
        t_embedder = object()
        dec_net = object()

    class DummyPatcher:
        diffusion_model = DummyInner()

    inner, path = locate_zimage_inner_model(DummyPatcher())
    assert inner is None
    assert path is None


def test_supported_sampler_guard_is_explicit() -> None:
    assert _SUPPORTED_SINGLE_EVAL_SAMPLERS == {"sample_euler"}


def test_forecast_sanitizer_clamps_and_reports() -> None:
    feature = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0e20], dtype=torch.float32)
    stats = _forecast_feature_sanitization_stats(feature, torch.float16)
    assert stats is not None
    sanitized = _sanitize_forecast_feature_for_final_layer(feature, torch.float16)
    assert sanitized.dtype == torch.float16
    assert torch.isfinite(sanitized).all()


def run_all() -> None:
    current = globals()
    tests = [current[name] for name in sorted(current) if name.startswith("test_")]
    for fn in tests:
        fn()
    print(f"ok: {len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
