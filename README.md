# KK Camera for ComfyUI

**KK Camera** 是一个面向 ComfyUI 的高级相机质感节点，用于给生成图、摄影后期图和电商视觉图添加更接近真实镜头、传感器与胶片成像的有机颗粒。

它不是简单叠加一层均匀噪声，而是根据画面亮度、暗部、高光与局部细节动态塑造颗粒密度，并同时提供微对比、整体对比、暖调和暗角控制。目标是让图像拥有更稳定、更克制、更像真实相机输出的“质感完成度”。

## Highlights

- **Optical-style organic grain**: 颗粒会随明暗区域变化，暗部更有呼吸感，高光保持干净。
- **Tone-aware rendering**: 暗部颗粒、高光颗粒和彩色噪点可独立控制，避免一键噪声带来的脏感。
- **Camera finishing controls**: 内置微对比、整体对比、暖调和暗角，适合做最后一层成片质感。
- **Pure Torch implementation**: 不引入额外 Python 依赖，直接复用 ComfyUI 环境里的 `torch`。
- **Deterministic seed**: 支持固定随机种子，便于工作流复现；设为 `-1` 时每次随机。
- **Bilingual-friendly node naming**: 节点显示为 `kk仿相机颗粒`，适合中文 ComfyUI 工作流。

## What It Does

这个节点更适合放在工作流末端，用来统一画面的“相机感”和“成片感”：

- 给 AI 生成图增加真实拍摄后的细腻颗粒，减少过度平滑的数字感。
- 给人像、产品、街拍、杂志风图像增加高级相机输出质感。
- 在不破坏主体清晰度的前提下，提升局部解析力、暗部层次和画面完整度。
- 为系列图提供一致的后期风格，可通过固定 seed 保持颗粒稳定。

## Installation

Clone or copy this folder into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tofu952711/comfyui-KK-camera-noise-.git
```

Then restart ComfyUI.

If you install manually, the final path should look like this:

```text
ComfyUI/custom_nodes/comfyui-KK-camera-noise-
```

No extra dependencies are required beyond a standard ComfyUI installation.

## Node

Add the node from:

```text
图像/后期处理 -> kk仿相机颗粒
```

Inputs and outputs:

| Port | Type | Description |
| --- | --- | --- |
| 图像 | IMAGE | Input image batch |
| 图像 | IMAGE | Processed image batch |

## Parameters

| Parameter | Suggested Range | Description |
| --- | ---: | --- |
| 颗粒强度 | `0.18 - 0.45` | Overall grain visibility. Lower values are clean and premium; higher values move toward film texture. |
| 颗粒大小 | `1.1 - 2.4` | Grain scale. Smaller values feel like high-end digital sensors; larger values feel more filmic. |
| 彩色噪点 | `0.05 - 0.18` | Chroma grain ratio. Keep this low for a refined camera look. |
| 暗部颗粒 | `0.35 - 0.85` | Grain weight in shadows. Raising this makes dark areas more organic. |
| 高光颗粒 | `0.08 - 0.25` | Grain weight in highlights. Keep restrained to avoid dirty highlights. |
| 微对比 | `0.08 - 0.24` | Local detail contrast, similar to a subtle clarity pass. |
| 整体对比 | `0.00 - 0.08` | Global contrast finishing. Negative values soften the image. |
| 暖调 | `0.01 - 0.06` | Warm lens/color response. Negative values create a cooler rendering. |
| 暗角 | `0.03 - 0.12` | Soft lens vignette. Use lightly for premium camera finishing. |
| 随机种子 | `42` / `-1` | Fixed seed for repeatable grain. `-1` randomizes each run. |

## Presets

### Quiet Leica-Inspired Finish

```text
颗粒强度: 0.24
颗粒大小: 1.6
彩色噪点: 0.10
暗部颗粒: 0.55
高光颗粒: 0.16
微对比: 0.14
整体对比: 0.04
暖调: 0.03
暗角: 0.07
随机种子: 42
```

### Clean Digital Magazine

```text
颗粒强度: 0.16
颗粒大小: 1.1
彩色噪点: 0.05
暗部颗粒: 0.35
高光颗粒: 0.10
微对比: 0.22
整体对比: 0.06
暖调: 0.01
暗角: 0.04
随机种子: 42
```

### Film Street Look

```text
颗粒强度: 0.42
颗粒大小: 2.3
彩色噪点: 0.16
暗部颗粒: 0.80
高光颗粒: 0.24
微对比: 0.12
整体对比: 0.03
暖调: 0.05
暗角: 0.11
随机种子: 42
```

## Recommended Workflow Position

Place `kk仿相机颗粒` near the end of the graph:

```text
Generated / upscaled image
-> color correction
-> sharpening or detail refinement
-> kk仿相机颗粒
-> save image
```

For best results, apply it after resizing/upscaling. Grain added before a major resize may become softened or distorted.

## Implementation Notes

The node works on ComfyUI `IMAGE` tensors in NHWC format and keeps the output in the same format. Internally it:

- builds multi-scale luminance grain with Torch random fields and interpolation;
- derives tonal masks from perceptual luma;
- applies local micro-contrast with a Gaussian blur detail pass;
- adds low-ratio chroma grain only when requested;
- applies a soft vignette mask as the final optical finish.

Everything runs in-process with Torch and can use the same device as the input tensor.

## License

MIT License. See [LICENSE](LICENSE).
