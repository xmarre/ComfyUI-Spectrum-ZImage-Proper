from __future__ import annotations

from .comfyui_spectrum_zimage import SpectrumConfig, ZImageSpectrumPatcher


class SpectrumApplyZImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 32, "step": 1}),
                "tail_actual_steps": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1}),
                "max_history": ("INT", {"default": 128, "min": 17, "max": 512, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "sampling/spectrum"

    def apply(
        self,
        model,
        enabled,
        blend_weight,
        degree,
        ridge_lambda,
        window_size,
        flex_window,
        warmup_steps,
        tail_actual_steps,
        max_history,
        debug,
    ):
        if not enabled:
            return (model,)

        cfg = SpectrumConfig(
            enabled=bool(enabled),
            backend="zimage",
            blend_weight=float(blend_weight),
            degree=int(degree),
            ridge_lambda=float(ridge_lambda),
            window_size=float(window_size),
            flex_window=float(flex_window),
            warmup_steps=int(warmup_steps),
            tail_actual_steps=int(tail_actual_steps),
            max_history=int(max_history),
            debug=bool(debug),
        ).validate()
        return (ZImageSpectrumPatcher.patch(model, cfg),)


NODE_CLASS_MAPPINGS = {
    "SpectrumApplyZImage": SpectrumApplyZImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumApplyZImage": "Spectrum Apply Z-Image",
}
