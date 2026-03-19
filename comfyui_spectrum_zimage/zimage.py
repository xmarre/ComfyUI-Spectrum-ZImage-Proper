from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from .config import SpectrumConfig
from .runtime import SpectrumRuntime

LOG = logging.getLogger(__name__)
_SUPPORTED_SINGLE_EVAL_SAMPLERS = frozenset({"sample_euler"})


def _clone_model(model: Any) -> Any:
    return model.clone() if hasattr(model, "clone") else model


def _ensure_model_options(model: Any) -> Dict[str, Any]:
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    return model.model_options


def _ensure_transformer_options(model: Any) -> Dict[str, Any]:
    options = _ensure_model_options(model)
    if "transformer_options" not in options or options["transformer_options"] is None:
        options["transformer_options"] = {}
    return options["transformer_options"]


def _is_current_comfyui_zimage_inner(inner: Any, owner: Any = None) -> bool:
    if inner is None:
        return False
    required = ("patchify_and_embed", "unpatchify", "layers", "final_layer", "t_embedder")
    if not all(hasattr(inner, attr) for attr in required):
        return False
    if hasattr(inner, "dec_net"):
        return False

    model_config = getattr(owner, "model_config", None)
    unet_config = getattr(model_config, "unet_config", None)
    if isinstance(unet_config, dict) and unet_config.get("z_image_modulation", False):
        return True

    final_layer = getattr(inner, "final_layer", None)
    mod = getattr(final_layer, "adaLN_modulation", None)
    if mod is None:
        return False
    try:
        linear = mod[-1]
        in_features = getattr(linear, "in_features", None)
    except Exception:
        in_features = None
    return in_features == 256


def locate_zimage_inner_model(model: Any) -> Tuple[Optional[Any], Optional[str]]:
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        inner = outer.diffusion_model
        if _is_current_comfyui_zimage_inner(inner, owner=outer):
            return inner, "model.diffusion_model"
    if hasattr(model, "diffusion_model"):
        inner = model.diffusion_model
        if _is_current_comfyui_zimage_inner(inner, owner=model):
            return inner, "diffusion_model"
    return None, None


def _sampler_name(sampler: Any) -> str:
    fn = getattr(sampler, "sampler_function", None)
    return getattr(fn, "__name__", type(sampler).__name__)


def _sanitize_forecast_feature_for_final_layer(feature: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not dtype.is_floating_point:
        return feature.to(dtype)
    finfo = torch.finfo(dtype)
    feature = feature.to(torch.float32)
    feature = torch.nan_to_num(feature, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    feature = feature.clamp(min=finfo.min, max=finfo.max)
    return feature.to(dtype)


def _forecast_feature_sanitization_stats(feature: torch.Tensor, dtype: torch.dtype) -> Optional[Dict[str, Any]]:
    if not dtype.is_floating_point or feature.numel() == 0:
        return None
    finfo = torch.finfo(dtype)
    feature_fp32 = feature.detach().to(torch.float32)
    finite_mask = torch.isfinite(feature_fp32)
    had_nonfinite = bool((~torch.isfinite(feature_fp32)).any().item())
    out_of_range = bool(((feature_fp32 < finfo.min) | (feature_fp32 > finfo.max)).any().item())
    if not had_nonfinite and not out_of_range:
        return None
    if bool(finite_mask.any().item()):
        finite_feature = feature_fp32[finite_mask]
        before_min = float(finite_feature.amin().item())
        before_max = float(finite_feature.amax().item())
    else:
        before_min = float("nan")
        before_max = float("nan")
    return {
        "target_dtype": str(dtype),
        "had_nonfinite": had_nonfinite,
        "out_of_range": out_of_range,
        "before_min": before_min,
        "before_max": before_max,
    }


def _signature_seq(values) -> tuple[Any, ...]:
    if values is None:
        return ()
    out = []
    for value in values:
        if isinstance(value, (str, int, float, bool, type(None))):
            out.append(value)
        else:
            out.append(repr(value))
    return tuple(out)


def _patches_signature(transformer_options: Dict[str, Any]) -> Optional[tuple[Any, ...]]:
    patches = transformer_options.get("patches")
    if not isinstance(patches, dict) or not patches:
        return None
    signature = []
    for key in sorted(patches.keys(), key=str):
        signature.append((str(key), len(patches[key]) if hasattr(patches[key], "__len__") else None))
    return tuple(signature)


def _build_branch_signature(transformer_options: Dict[str, Any]) -> Optional[tuple[Any, ...]]:
    signature = []
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    if cond_or_uncond is not None:
        try:
            signature.append(("cond_or_uncond", tuple(int(v) for v in cond_or_uncond)))
        except Exception:
            signature.append(("cond_or_uncond", tuple(cond_or_uncond)))

    uuids = transformer_options.get("uuids")
    normalized_uuids = _signature_seq(uuids)
    if normalized_uuids:
        signature.append(("uuids", normalized_uuids))

    patches_sig = _patches_signature(transformer_options)
    if patches_sig is not None:
        signature.append(("patches", patches_sig))

    if not signature:
        return None
    return tuple(signature)


def _extract_step_context(transformer_options: Dict[str, Any]) -> Optional[tuple[SpectrumRuntime, int, int, bool]]:
    runtime = transformer_options.get("spectrum_runtime")
    run_id = transformer_options.get("spectrum_run_id")
    solver_step_id = transformer_options.get("spectrum_solver_step_id")
    actual_forward = transformer_options.get("spectrum_actual_forward")
    if runtime is None or run_id is None or solver_step_id is None or actual_forward is None:
        return None
    return runtime, int(run_id), int(solver_step_id), bool(actual_forward)


def _runtime_from_model_options(model_options: Dict[str, Any]) -> Optional[SpectrumRuntime]:
    transformer_options = (model_options or {}).get("transformer_options") or {}
    runtime = transformer_options.get("spectrum_runtime")
    if isinstance(runtime, SpectrumRuntime):
        return runtime
    return None


def _copy_model_options_with_step_context(
    model_options: Dict[str, Any], runtime: SpectrumRuntime, decision: Dict[str, Any]
) -> Dict[str, Any]:
    patched_model_options = dict(model_options or {})
    transformer_options = dict(patched_model_options.get("transformer_options") or {})
    patched_model_options["transformer_options"] = transformer_options
    transformer_options["spectrum_runtime"] = runtime
    transformer_options["spectrum_run_id"] = decision["run_id"]
    transformer_options["spectrum_solver_step_id"] = decision["solver_step_id"]
    transformer_options["spectrum_time_coord"] = decision["time_coord"]
    transformer_options["spectrum_actual_forward"] = decision["actual_forward"]
    transformer_options["spectrum_step_finalized"] = False
    return patched_model_options


def _install_sampler_level_wrappers(model: Any, runtime: SpectrumRuntime) -> None:
    options = _ensure_model_options(model)
    wrapper_state = options.get("_spectrum_sampler_wrapper_state")
    if not isinstance(wrapper_state, dict):
        wrapper_state = {}
        options["_spectrum_sampler_wrapper_state"] = wrapper_state
    wrapper_state["runtime"] = runtime
    if options.get("_spectrum_sampler_wrappers_installed", False):
        return

    import comfy.patcher_extension

    def outer_sample_wrapper(
        executor,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        latent_shapes=None,
    ):
        wrapper_runtime = wrapper_state.get("runtime")
        if not isinstance(wrapper_runtime, SpectrumRuntime):
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        sampler_name = _sampler_name(sampler)
        supports_solver_steps = sampler_name in _SUPPORTED_SINGLE_EVAL_SAMPLERS
        run_id = wrapper_runtime.start_run(sigmas, sampler_name, supports_solver_steps=supports_solver_steps)
        try:
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            if wrapper_runtime.cfg.debug:
                stats = wrapper_runtime.stats
                LOG.warning(
                    "Spectrum run_id=%s sampler=%s total_steps=%s actual=%s forecast=%s disabled=%s reason=%r",
                    stats.run_id,
                    stats.sampler_name,
                    stats.total_steps,
                    stats.actual_forward_count,
                    stats.forecasted_count,
                    stats.forecast_disabled,
                    stats.disable_reason,
                )
            wrapper_runtime.end_run(run_id)

    def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
        effective_model_options = model_options or {}
        wrapper_runtime = _runtime_from_model_options(effective_model_options)
        if wrapper_runtime is None:
            wrapper_runtime = wrapper_state.get("runtime")
        if not isinstance(wrapper_runtime, SpectrumRuntime):
            return executor(x, timestep, effective_model_options, seed)
        if wrapper_runtime.active_run_id is None:
            return executor(x, timestep, effective_model_options, seed)
        if not wrapper_runtime.active_run_supports_solver_steps:
            return executor(x, timestep, effective_model_options, seed)

        step_id = wrapper_runtime.next_solver_step_id
        total_steps = wrapper_runtime.num_steps()
        time_coord = wrapper_runtime.time_coord_for_step(step_id)
        decision = wrapper_runtime.begin_solver_step(wrapper_runtime.active_run_id, step_id, time_coord, total_steps)
        patched_model_options = _copy_model_options_with_step_context(
            effective_model_options, wrapper_runtime, decision
        )
        try:
            return executor(x, timestep, patched_model_options, seed)
        finally:
            wrapper_runtime.finalize_solver_step(
                decision["run_id"],
                decision["solver_step_id"],
                used_forecast=wrapper_runtime.step_used_forecast(
                    decision["run_id"], decision["solver_step_id"]
                ),
            )

    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        outer_sample_wrapper,
        options,
        is_model_options=True,
    )
    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        predict_noise_wrapper,
        options,
        is_model_options=True,
    )
    options["_spectrum_sampler_wrappers_installed"] = True


def _install_generic_zimage_wrapper(inner: Any) -> None:
    if getattr(inner, "_spectrum_zimage_forward_installed", False):
        return

    original_forward = inner._forward
    setattr(inner, "_spectrum_original__forward", original_forward)

    def spectrum_forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        ref_latents=None,
        ref_contexts=None,
        siglip_feats=None,
        transformer_options=None,
        **kwargs,
    ):
        options = transformer_options or {}
        runtime = options.get("spectrum_runtime") if isinstance(options, dict) else None
        if runtime is None or not getattr(runtime.cfg, "enabled", False):
            return self._spectrum_original__forward(
                x,
                timesteps,
                context,
                num_tokens,
                attention_mask=attention_mask,
                ref_latents=[] if ref_latents is None else ref_latents,
                ref_contexts=[] if ref_contexts is None else ref_contexts,
                siglip_feats=[] if siglip_feats is None else siglip_feats,
                transformer_options=options,
                **kwargs,
            )

        return _run_zimage_forward_with_spectrum(
            self,
            runtime,
            x=x,
            timesteps=timesteps,
            context=context,
            num_tokens=num_tokens,
            attention_mask=attention_mask,
            ref_latents=[] if ref_latents is None else ref_latents,
            ref_contexts=[] if ref_contexts is None else ref_contexts,
            siglip_feats=[] if siglip_feats is None else siglip_feats,
            transformer_options=options,
            **kwargs,
        )

    inner._forward = spectrum_forward.__get__(inner, type(inner))
    setattr(inner, "_spectrum_zimage_forward_installed", True)


def _run_zimage_forward_with_spectrum(
    inner: Any,
    runtime: SpectrumRuntime,
    *,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    num_tokens,
    attention_mask: Optional[torch.Tensor],
    ref_latents,
    ref_contexts,
    siglip_feats,
    transformer_options: Dict[str, Any],
    **kwargs,
) -> torch.Tensor:
    import comfy.ldm.common_dit

    transformer_options = (transformer_options or {}).copy()
    patches = transformer_options.get("patches", {})
    step_ctx = _extract_step_context(transformer_options)
    run_id: Optional[int] = None
    solver_step_id: Optional[int] = None

    ref_latents = [] if ref_latents is None else ref_latents
    ref_contexts = [] if ref_contexts is None else ref_contexts
    siglip_feats = [] if siglip_feats is None else siglip_feats

    omni = len(ref_latents) > 0
    if omni:
        timesteps = torch.cat([timesteps * 0, timesteps], dim=0)

    t = 1.0 - timesteps
    cap_feats = context
    cap_mask = attention_mask
    _bs, _c, h, w = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (inner.patch_size, inner.patch_size))

    t = inner.t_embedder(t * inner.time_scale, dtype=x.dtype)
    adaln_input = t

    if inner.clip_text_pooled_proj is not None:
        pooled = kwargs.get("clip_text_pooled", None)
        if pooled is not None:
            pooled = inner.clip_text_pooled_proj(pooled)
        else:
            pooled = torch.zeros((x.shape[0], inner.clip_text_dim), device=x.device, dtype=x.dtype)
        adaln_input = inner.time_text_embed(torch.cat((t, pooled), dim=-1))

    x_is_tensor = isinstance(x, torch.Tensor)
    img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = inner.patchify_and_embed(
        x,
        cap_feats,
        cap_mask,
        adaln_input,
        num_tokens,
        ref_latents=ref_latents,
        ref_contexts=ref_contexts,
        siglip_feats=siglip_feats,
        transformer_options=transformer_options,
    )
    freqs_cis = freqs_cis.to(img.device)
    expected_feature_shape = tuple(img.shape)
    branch_signature = _build_branch_signature(transformer_options)

    if step_ctx is not None:
        _, run_id, solver_step_id, actual_forward = step_ctx
        runtime.register_model_hook_call(
            run_id,
            solver_step_id,
            expected_shape=expected_feature_shape,
            branch_signature=branch_signature,
        )
        if not actual_forward:
            pred_feature = runtime.predict_feature(
                run_id,
                solver_step_id,
                expected_shape=expected_feature_shape,
                branch_signature=branch_signature,
            )
            if pred_feature is not None:
                if runtime.cfg.debug:
                    sanitize_stats = _forecast_feature_sanitization_stats(pred_feature, img.dtype)
                    if sanitize_stats is not None:
                        LOG.warning(
                            "Spectrum sanitized forecast run_id=%s step=%s target_dtype=%s had_nonfinite=%s out_of_range=%s before_min=%s before_max=%s",
                            run_id,
                            solver_step_id,
                            sanitize_stats["target_dtype"],
                            sanitize_stats["had_nonfinite"],
                            sanitize_stats["out_of_range"],
                            sanitize_stats["before_min"],
                            sanitize_stats["before_max"],
                        )
                pred_feature = _sanitize_forecast_feature_for_final_layer(pred_feature, img.dtype)
                pred_feature = inner.final_layer(pred_feature, adaln_input, timestep_zero_index=timestep_zero_index)
                pred_feature = inner.unpatchify(pred_feature, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]
                return -pred_feature

    transformer_options["total_blocks"] = len(inner.layers)
    transformer_options["block_type"] = "double"
    img_input = img

    for block_index, layer in enumerate(inner.layers):
        transformer_options["block_index"] = block_index
        img = layer(
            img,
            mask,
            freqs_cis,
            adaln_input,
            timestep_zero_index=timestep_zero_index,
            transformer_options=transformer_options,
        )
        if "double_block" in patches:
            for patch in patches["double_block"]:
                out = patch({
                    "img": img[:, cap_size[0]:],
                    "img_input": img_input[:, cap_size[0]:],
                    "txt": img[:, :cap_size[0]],
                    "pe": freqs_cis[:, cap_size[0]:],
                    "vec": adaln_input,
                    "x": x,
                    "block_index": block_index,
                    "transformer_options": transformer_options,
                })
                if "img" in out:
                    img[:, cap_size[0]:] = out["img"]
                if "txt" in out:
                    img[:, :cap_size[0]] = out["txt"]

    if run_id is not None and solver_step_id is not None:
        runtime.observe_actual_feature_for_branch(
            run_id,
            solver_step_id,
            img,
            branch_signature=branch_signature,
        )

    img = inner.final_layer(img, adaln_input, timestep_zero_index=timestep_zero_index)
    img = inner.unpatchify(img, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]
    return -img


class ZImageSpectrumPatcher:
    @staticmethod
    def patch(model: Any, cfg: SpectrumConfig) -> Any:
        cfg = cfg.validate()
        patched = _clone_model(model)
        transformer_options = _ensure_transformer_options(patched)
        runtime = SpectrumRuntime(cfg)
        transformer_options["spectrum_runtime"] = runtime
        transformer_options["spectrum_enabled"] = cfg.enabled
        transformer_options["spectrum_backend"] = "zimage"
        transformer_options["spectrum_cfg"] = cfg.to_dict()
        _install_sampler_level_wrappers(patched, runtime)

        inner, path = locate_zimage_inner_model(patched)
        if inner is None:
            raise RuntimeError(
                "Could not locate the current ComfyUI Z-Image diffusion model. This node currently supports native Z-Image models on the lumina/NextDiT path only."
            )
        _install_generic_zimage_wrapper(inner)
        if cfg.debug:
            LOG.warning("Spectrum installed on %s", path)
        return patched
