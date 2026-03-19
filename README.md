# ComfyUI-Spectrum-ZImage-Proper

Faithful **ComfyUI Z-Image** port of [**Spectrum**](https://hanjq17.github.io/Spectrum/) from *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*.

This repo is intentionally narrow in scope: it patches the **current native ComfyUI Z-Image path** instead of pretending one generic wrapper can safely cover every backend.

## What this node does

`Spectrum Apply Z-Image` patches the native ComfyUI Z-Image diffusion model on the **MODEL** path and applies Spectrum-style forecasting to the **full unified hidden token sequence immediately before `final_layer`**.

That is the practical Z-Image equivalent of the paper's "last-block-only" strategy:

1. run a real Z-Image forward on selected steps
2. cache the final hidden sequence after the main transformer stack
3. fit a small Chebyshev ridge regressor online over the detected solver-step coordinate
4. forecast future hidden sequences on skipped steps
5. still apply the real `final_layer` and real unpatchify path for the current conditioning

## Why the hook is here

For current ComfyUI Z-Image support, the native backend runs a Lumina/NextDiT-style path where `patchify_and_embed(...)` produces a unified sequence, `self.layers` updates it, and `self.final_layer(...)` is applied afterward before unpatchifying the image result. This repo patches that exact pre-head sequence instead of forecasting the final denoised image tensor.

That keeps the integration aligned with the core Spectrum idea: forecast a model-internal hidden feature at the backend-specific integration point, not the post-head output.

## Current scope

Supported:

- native **ComfyUI Z-Image** models on the current lumina/NextDiT path
- Z-Image and Z-Image-Turbo on that same backend path
- normal MODEL-path LoRA usage
- standard `transformer_options` patch chains

Potentially usable but not promised:

- other Z-Image variants that still resolve to the same current backend path and hidden-shape contract

Not included:

- non-Z-Image Lumina-family models
- pixel-space decoder backends
- multi-eval-per-step samplers

## Installation

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-Spectrum-ZImage-Proper
```

Restart ComfyUI.

No extra Python dependencies are required beyond what ComfyUI already provides.

## Node

### Spectrum Apply Z-Image

**Input:** `MODEL`

**Output:** `MODEL`

Place it on the Z-Image model line:

```text
Model loader -> LoRA stack -> Spectrum Apply Z-Image -> guider / sampler
```

Recommended placement:

- after model loading and LoRA application
- before guider/sampler nodes

## Parameters

### `blend_weight`

Blend between linear local extrapolation and Chebyshev spectral prediction.

- `1.0` = pure spectral predictor
- `0.0` = pure local linear predictor
- recommended default: `0.5`

### `degree`

Chebyshev degree.

Recommended default: `4`

### `ridge_lambda`

Ridge regularization for the online coefficient fit.

Recommended default: `0.1`

### `window_size`

Initial interval size before a real forward is required again.

Recommended default: `2.0`

### `flex_window`

How much the interval grows after each post-warmup real forward.

This is a practical ComfyUI-facing adaptive schedule, not the paper's exact `alpha/N/W` scheduler parameterization.

- `0.75` = moderate speedup setting
- `3.0` = more aggressive setting

### `warmup_steps`

Number of initial real forwards before forecasting is allowed.

Recommended default: `5`

### `tail_actual_steps`

Number of final solver steps forced to stay on the real path.

Practical default: `3`

### `max_history`

Cap for cached real-forward feature points used for the fit.

### `debug`

Enables lightweight logging during patch install and a per-run summary of actual vs forecasted solver steps.

## Recommended settings

### Safer / closer to the paper's moderate setting

- `blend_weight = 0.50`
- `degree = 4`
- `ridge_lambda = 0.10`
- `window_size = 2.0`
- `flex_window = 0.75`
- `warmup_steps = 5`
- `tail_actual_steps = 3`

### More aggressive

- `blend_weight = 0.75`
- `degree = 4`
- `ridge_lambda = 0.10`
- `window_size = 2.0`
- `flex_window = 3.0`
- `warmup_steps = 5`
- `tail_actual_steps = 3`

## Design notes

### 1. Forecast target is the unified pre-head Z-Image hidden sequence

This repo caches and forecasts the full hidden token sequence **after the main `self.layers` stack and before `final_layer`**.

That is the Z-Image-specific equivalent of forecasting the final hidden feature at the model's backend-specific integration point. Forecasting the final denoised image tensor directly would be less faithful and would bypass the current backend head structure.

### 2. Runtime state is per patched model clone, not per globally patched inner model

ComfyUI model clones can share the same underlying diffusion-model object. If the monkey-patch closes over one runtime object, forecast state can leak between clones.

This repo avoids that by:

- patching the inner Z-Image model only once
- storing the active runtime in each cloned model's `transformer_options`
- looking up the runtime dynamically on every call
- falling back to the original `_forward` when Spectrum is not active

### 3. Step normalization uses detected schedule coordinates

The paper mostly benchmarks 50-step runs. ComfyUI users do not.

This repo normalizes the predictor against the detected solver-step coordinates from the active sigma schedule instead of hard-coding a 50-step assumption.

## Validation / smoke test

A lightweight import/runtime smoke test is included:

```bash
python tests/smoke_runtime.py
```

## Known limitations

- This repo currently targets the **native current ComfyUI Z-Image lumina/NextDiT backend path** only.
- Forecasting is only enabled for deterministic `sample_euler`; samplers that do not preserve a one-`predict_noise`-per-solver-step contract are treated as unsupported and fall back to real forwards.
- The adaptive schedule exposed here is a practical window-growth scheduler, not the paper's exact `alpha/N/W` interface. The Chebyshev+ridges part is faithful; the scheduler surface is an implementation approximation carried over from the existing ComfyUI ports.
- The node still executes `patchify_and_embed`, conditioning refiners, `final_layer`, and unpatchify on every step. The skipped work is the expensive main transformer stack (`self.layers`).
- Per-layer `patches["double_block"]` hooks on the main stack are only applied on actual-forward steps. Forecasted steps bypass that stack by design.
- If another custom node fully replaces the same Z-Image `_forward` implementation after this node has patched it, compatibility is not guaranteed.
- This repo assumes the hidden-token shape for a given branch/signature stays stable across a run. If that invariant is violated, forecasting is disabled for safety.

## Assumptions

- You are using a ComfyUI build with native Z-Image support on the current lumina/NextDiT backend.
- The loaded model resolves to the `z_image_modulation` path or an equivalent current backend exposing the same `patchify_and_embed -> layers -> final_layer -> unpatchify` contract.
- You want a training-free inference-time acceleration patch, not a distilled checkpoint or a new sampler.
