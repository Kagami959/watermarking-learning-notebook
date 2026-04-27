# attacks.py 代码逐行分析

> 文件来源：`D:\C\Python\backdoor\audioseal\examples\attacks.py`
> 项目：Meta AudioSeal（音频水印工具）
> 用途：音频攻击效果集合，用于测试水印鲁棒性

---

## 1. 版权与导入

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 第 1-9 行为 MIT 许可证声明及说明注释
```

```python
import typing as tp      # 类型提示模块
import julius            # PyTorch 音频处理库（重采样、滤波、卷积）
import torch             # PyTorch 深度学习框架
```

---

## 2. 独立函数

### `generate_pink_noise(length)` — 生成粉红噪声

| 行号 | 代码 | 作用 |
|------|------|------|
| 22 | `num_rows = 16` | Voss-McCartney 算法使用的随机源行数 |
| 23 | `array = torch.randn(num_rows, length // num_rows + 1)` | 生成 16 行的正态随机矩阵 |
| 24 | `reshaped_array = torch.cumsum(array, dim=1)` | 沿列方向累积求和，模拟不同更新频率 |
| 25 | `reshaped_array = reshaped_array.reshape(-1)` | 展平为一维向量 |
| 26 | `reshaped_array = reshaped_array[:length]` | 截取到精确长度 |
| 28 | `pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))` | 归一化到 [-1, 1] |

### `audio_effect_return(tensor, mask)` — 统一返回格式

| 行号 | 代码 | 作用 |
|------|------|------|
| 36-37 | `if mask is None: return tensor` | 无 mask 时只返回张量 |
| 38-39 | `else: return tensor, mask` | 有 mask 时返回元组 |

---

## 3. AudioEffects 类（所有攻击方法）

所有方法均为 `@staticmethod`，接收音频张量和可选的 mask，返回处理后的张量。

### 3.1 `speed()` — 变速攻击（第 43-69 行）

**原理**：通过重采样改变播放速度。

| 行号 | 代码 | 作用 |
|------|------|------|
| 61 | `speed = torch.FloatTensor(1).uniform_(*speed_range)` | 从 (0.5, 1.5) 随机采样速度值 |
| 62 | `new_sr = int(sample_rate * 1 / speed)` | 计算目标采样率（速度越快采样率越低） |
| 63 | `resampled_tensor = julius.resample_frac(tensor, sample_rate, new_sr)` | 重采样实现变速 |
| 64-69 | `if mask is None ... else ...` | mask 用最近邻插值同步缩放 |

**参数**：`speed_range=(0.5, 1.5)`，`sample_rate=16000`

---

### 3.2 `updownresample()` — 升降采样攻击（第 71-86 行）

**原理**：先升采样再降采样，引入插值精度损失。

| 行号 | 代码 | 作用 |
|------|------|------|
| 79 | `orig_shape = tensor.shape` | 保存原始形状 |
| 81 | `tensor = julius.resample_frac(tensor, sample_rate, intermediate_freq)` | 升采样：16kHz → 32kHz |
| 83 | `tensor = julius.resample_frac(tensor, intermediate_freq, sample_rate)` | 降采样：32kHz → 16kHz |
| 85 | `assert tensor.shape == orig_shape` | 断言形状不变 |

**参数**：`sample_rate=16000`，`intermediate_freq=32000`

---

### 3.3 `echo()` — 回声/混响攻击（第 88-139 行）

**原理**：构造脉冲响应（直达声 + 延迟反射声），与音频卷积。

| 行号 | 代码 | 作用 |
|------|------|------|
| 108 | `duration = torch.FloatTensor(1).uniform_(*duration_range)` | 随机持续时间 (0.1~0.5s) |
| 109 | `volume = torch.FloatTensor(1).uniform_(*volume_range)` | 随机回声音量 (0.1~0.5) |
| 111 | `n_samples = int(sample_rate * duration)` | 计算脉冲响应样本数 |
| 112 | `impulse_response = torch.zeros(...)` | 创建全零脉冲响应 |
| 115 | `impulse_response[0] = 1.0` | 第一个样本=1.0，代表直达声 |
| 117-119 | `impulse_response[...] = volume` | 末尾样本=回声音量，代表延迟反射 |
| 122 | `impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)` | 扩展为 [1, 1, n] 格式 |
| 125 | `reverbed_signal = julius.fft_conv1d(tensor, impulse_response)` | FFT 卷积产生混响 |
| 128-132 | `reverbed_signal = reverbed_signal / max * max(tensor)` | 归一化到原始振幅范围 |
| 135-137 | `tmp = torch.zeros_like(tensor)...` | 截断/填充回原始长度 |

**参数**：`volume_range=(0.1, 0.5)`，`duration_range=(0.1, 0.5)`，`sample_rate=16000`

---

### 3.4 `random_noise()` — 高斯白噪声攻击（第 141-150 行）

| 行号 | 代码 | 作用 |
|------|------|------|
| 148 | `noise = torch.randn_like(waveform) * noise_std` | 生成高斯白噪声 |
| 149 | `noisy_waveform = waveform + noise` | 叠加到原始音频 |

**参数**：`noise_std=0.001`

---

### 3.5 `pink_noise()` — 粉红噪声攻击（第 152-163 行）

| 行号 | 代码 | 作用 |
|------|------|------|
| 159 | `noise = generate_pink_noise(waveform.shape[-1]) * noise_std` | 生成粉红噪声并缩放 |
| 160 | `noise = noise.to(waveform.device)` | 移到相同设备 |
| 162 | `noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0)` | 扩展维度后叠加 |

**参数**：`noise_std=0.01`

---

### 3.6 `lowpass_filter()` — 低通滤波攻击（第 165-176 行）

| 行号 | 代码 | 作用 |
|------|------|------|
| 173-174 | `julius.lowpass_filter(waveform, cutoff=cutoff_freq / sample_rate)` | 滤除高于截止频率的成分 |

**参数**：`cutoff_freq=5000`，`sample_rate=16000`

---

### 3.7 `highpass_filter()` — 高通滤波攻击（第 178-189 行）

| 行号 | 代码 | 作用 |
|------|------|------|
| 186-187 | `julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate)` | 滤除低于截止频率的成分 |

**参数**：`cutoff_freq=500`，`sample_rate=16000`

---

### 3.8 `bandpass_filter()` — 带通滤波攻击（第 191-220 行）

| 行号 | 代码 | 作用 |
|------|------|------|
| 213-218 | `julius.bandpass_filter(waveform, cutoff_low=..., cutoff_high=...)` | 只保留中间频段 |

**参数**：`cutoff_freq_low=300`，`cutoff_freq_high=8000`，`sample_rate=16000`

---

### 3.9 `smooth()` — 平滑攻击（第 222-250 行）

| 行号 | 代码 | 作用 |
|------|------|------|
| 239 | `window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))` | 随机窗口大小 (2~10) |
| 241 | `kernel = torch.ones(...) / window_size` | 创建均值滤波核 |
| 244 | `smoothed = julius.fft_conv1d(tensor, kernel)` | FFT 卷积实现滑动平均 |
| 246-248 | `tmp = torch.zeros_like(tensor)...` | 截断回原始长度 |

**参数**：`window_size_range=(2, 10)`

---

### 3.10 `boost_audio()` — 增强音量（第 252-258 行）

```python
return audio_effect_return(tensor=tensor * (1 + amount / 100), mask=mask)
```

- 振幅放大 `amount%`（默认 +20%，即 ×1.2）

---

### 3.11 `duck_audio()` — 降低音量（第 260-266 行）

```python
return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)
```

- 振幅缩小 `amount%`（默认 -20%，即 ×0.8）

---

### 3.12 `identity()` — 恒等变换（第 268-272 行）

```python
return audio_effect_return(tensor=tensor, mask=mask)
```

- 不做任何处理，作为 baseline 对照组

---

### 3.13 `shush()` — 部分静音（第 274-296 行）

| 行号 | 代码 | 作用 |
|------|------|------|
| 290 | `time = tensor.size(-1)` | 获取时间维度长度 |
| 291 | `shush_tensor = tensor.detach().clone()` | 克隆张量（断开计算图） |
| 294 | `shush_tensor[:, :, :int(fraction*time)] = 0.0` | 开头 `fraction` 比例置零 |

**参数**：`fraction=0.001`（静音前 0.1% 的样本）

---

## 4. 模块总结

| 方法 | 攻击类型 | 模拟场景 |
|------|----------|----------|
| `speed()` | 变速 | 播放速度变化 |
| `updownresample()` | 重采样损失 | 编解码/传输精度损失 |
| `echo()` | 回声/混响 | 空间反射、混响效果 |
| `random_noise()` | 高斯白噪声 | 背景噪声干扰 |
| `pink_noise()` | 粉红噪声 | 自然环境噪声 |
| `lowpass_filter()` | 低通滤波 | 扬声器/电话带宽限制 |
| `highpass_filter()` | 高通滤波 | 去除低频隆隆声 |
| `bandpass_filter()` | 带通滤波 | 通信信道带宽限制 |
| `smooth()` | 滑动平均 | 信号平滑/模糊 |
| `boost_audio()` | 增幅 | 音量放大 |
| `duck_audio()` | 衰减 | 音量缩小 |
| `identity()` | 恒等 | 无攻击，基准对照 |
| `shush()` | 部分静音 | 开头截断/静音 |

## 5. 设计模式

- **统一接口**：所有方法接收 `(tensor, ..., mask)`，返回 `tensor` 或 `(tensor, mask)`
- **随机化参数**：`speed`、`echo`、`smooth` 的关键参数在给定范围内随机采样，增加攻击多样性
- **mask 机制**：可选 mask 用于标记水印嵌入区域，攻击时同步变换 mask
- **依赖库**：核心音频操作依赖 `julius`（重采样、滤波、FFT 卷积）
