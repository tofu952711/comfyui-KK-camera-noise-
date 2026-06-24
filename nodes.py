import math

import torch
import torch.nn.functional as F


def _as_float(value):
    return float(value)


def _make_generator(device, seed):
    if seed is None or int(seed) < 0:
        return None, device

    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return generator, device
    except Exception:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return generator, torch.device("cpu")


def _randn(shape, device, dtype, seed=None):
    generator, generator_device = _make_generator(device, seed)
    noise = torch.randn(shape, generator=generator, device=generator_device, dtype=dtype)
    return noise.to(device=device, dtype=dtype)


def _seed_offset(seed, offset):
    seed = int(seed)
    if seed < 0:
        return -1
    return (seed + int(offset)) & 0xFFFFFFFF


def _value(inputs, chinese_name, english_name, default=None):
    if chinese_name in inputs:
        return inputs[chinese_name]
    if english_name in inputs:
        return inputs[english_name]
    return default


def _odd_kernel_size(value):
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _gaussian_kernel(size, sigma, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def _blur_nhwc(image, radius):
    if radius <= 0:
        return image

    kernel_size = _odd_kernel_size(radius * 2 + 1)
    sigma = max(0.8, radius / 2)
    b, h, w, c = image.shape
    image_nchw = image.permute(0, 3, 1, 2)
    kernel = _gaussian_kernel(kernel_size, sigma, image.device, image.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(image_nchw, kernel, padding=padding, groups=c)
    return blurred.permute(0, 2, 3, 1)


def _grain_pattern(batch, height, width, channels, grain_size, device, dtype, seed):
    grain_size = max(1.0, float(grain_size))
    low_h = max(8, int(math.ceil(height / grain_size)))
    low_w = max(8, int(math.ceil(width / grain_size)))
    base = _randn((batch, low_h, low_w, channels), device, dtype, seed)

    if low_h == height and low_w == width:
        return base

    base = base.permute(0, 3, 1, 2)
    mode = "bicubic" if grain_size > 1.3 else "bilinear"
    upsampled = F.interpolate(base, size=(height, width), mode=mode, align_corners=False)
    return upsampled.permute(0, 2, 3, 1)


def _vignette_mask(batch, height, width, device, dtype):
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx * xx + yy * yy)
    mask = torch.clamp((radius - 0.25) / 0.95, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    return mask.view(1, height, width, 1).repeat(batch, 1, 1, 1)


class PremiumCameraOpticalGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "颗粒强度": (
                    "FLOAT",
                    {"default": 0.28, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "颗粒大小": (
                    "FLOAT",
                    {"default": 1.6, "min": 0.5, "max": 8.0, "step": 0.1},
                ),
                "彩色噪点": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "暗部颗粒": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "高光颗粒": (
                    "FLOAT",
                    {"default": 0.18, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "微对比": (
                    "FLOAT",
                    {"default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "整体对比": (
                    "FLOAT",
                    {"default": 0.05, "min": -0.5, "max": 0.5, "step": 0.01},
                ),
                "暖调": (
                    "FLOAT",
                    {"default": 0.03, "min": -0.3, "max": 0.3, "step": 0.01},
                ),
                "暗角": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
                "随机种子": ("INT", {"default": 42, "min": -1, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "apply"
    CATEGORY = "图像/后期处理"

    def apply(self, **inputs):
        image = _value(inputs, "图像", "image")
        grain_strength = _value(inputs, "颗粒强度", "grain_strength", 0.28)
        grain_size = _value(inputs, "颗粒大小", "grain_size", 1.6)
        color_noise = _value(inputs, "彩色噪点", "color_noise", 0.12)
        shadow_grain = _value(inputs, "暗部颗粒", "shadow_grain", 0.55)
        highlight_grain = _value(inputs, "高光颗粒", "highlight_grain", 0.18)
        micro_contrast = _value(inputs, "微对比", "micro_contrast", 0.16)
        global_contrast = _value(inputs, "整体对比", "global_contrast", 0.05)
        warm_tint = _value(inputs, "暖调", "warm_tint", 0.03)
        vignette = _value(inputs, "暗角", "vignette", 0.08)
        seed = _value(inputs, "随机种子", "seed", 42)

        source = image.clamp(0.0, 1.0)
        b, h, w, c = source.shape
        device = source.device
        dtype = source.dtype

        if c < 3:
            return (source,)

        rgb = source[..., :3]
        alpha_or_extra = source[..., 3:] if c > 3 else None

        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)
        luma = (rgb * luma_weights).sum(dim=-1, keepdim=True)

        if micro_contrast > 0:
            detail_radius = max(1, int(round(float(grain_size) * 2.5)))
            low_frequency = _blur_nhwc(rgb, detail_radius)
            rgb = (rgb + (rgb - low_frequency) * float(micro_contrast)).clamp(0.0, 1.0)
            luma = (rgb * luma_weights).sum(dim=-1, keepdim=True)

        if global_contrast != 0:
            contrast = 1.0 + float(global_contrast)
            rgb = ((rgb - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
            luma = (rgb * luma_weights).sum(dim=-1, keepdim=True)

        if warm_tint != 0:
            tint = float(warm_tint)
            tint_vector = torch.tensor([1.0 + tint, 1.0 + tint * 0.28, 1.0 - tint * 0.72], device=device, dtype=dtype)
            rgb = (rgb * tint_vector.view(1, 1, 1, 3)).clamp(0.0, 1.0)
            luma = (rgb * luma_weights).sum(dim=-1, keepdim=True)

        base_strength = float(grain_strength) * 0.035
        shadow_mask = torch.pow(1.0 - luma, 1.35) * float(shadow_grain)
        highlight_mask = torch.pow(luma, 1.9) * float(highlight_grain)
        tonal_mask = 0.42 + shadow_mask + highlight_mask

        luma_noise = _grain_pattern(b, h, w, 1, grain_size, device, dtype, _seed_offset(seed, 0))
        fine_noise = _grain_pattern(b, h, w, 1, max(0.7, float(grain_size) * 0.55), device, dtype, _seed_offset(seed, 17))
        organic_noise = luma_noise * 0.75 + fine_noise * 0.25

        rgb = rgb + organic_noise * tonal_mask * base_strength

        if color_noise > 0:
            chroma = _grain_pattern(b, h, w, 3, max(1.0, float(grain_size) * 1.35), device, dtype, _seed_offset(seed, 101))
            chroma = chroma - chroma.mean(dim=-1, keepdim=True)
            rgb = rgb + chroma * tonal_mask * base_strength * float(color_noise) * 0.9

        if vignette > 0:
            mask = _vignette_mask(b, h, w, device, dtype)
            rgb = rgb * (1.0 - mask * float(vignette))

        rgb = rgb.clamp(0.0, 1.0)
        if alpha_or_extra is not None:
            rgb = torch.cat([rgb, alpha_or_extra], dim=-1)

        return (rgb,)


class KKCameraGrain(PremiumCameraOpticalGrain):
    pass


NODE_CLASS_MAPPINGS = {
    "KKCameraGrain": KKCameraGrain,
    "PremiumCameraOpticalGrain": PremiumCameraOpticalGrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KKCameraGrain": "kk仿相机颗粒",
    "PremiumCameraOpticalGrain": "kk仿相机颗粒",
}
