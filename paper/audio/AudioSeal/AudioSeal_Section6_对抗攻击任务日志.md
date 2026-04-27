# AudioSeal Section 6 对抗性水印移除实验 — 任务日志

## 任务信息
- **日期**: 2026-04-13
- **目标**: 补全 AudioSeal 论文 Section 6 "Adversarial Watermark Removal" 的实验（白盒、半黑盒、黑盒三种对抗攻击）
- **论文**: AudioSeal: Proactive Detection of Voice Cloning with Localized Watermarking (arXiv:2401.17264)
- **实验路径**: `D:\C\Python\backdoor\audiosealexp\section6_adversarial\`
- **状态**: 未开始（计划已通过，待实现）

---

## 1. 当前进度

| 步骤 | 状态 | 备注 |
|------|------|------|
| Table 3 信号处理攻击实验 | ✅ 完成 | 13/15 攻击与论文一致 |
| Table 3 实验已推送 GitHub | ✅ 完成 | 在 `paper/audio/AudioSeal/experiment/` 下 |
| Section 6 计划制定 | ✅ 完成 | 计划文件已保存 |
| Section 6 代码实现 | ❌ 未开始 | 需要时继续 |
| Section 6 代理模型训练 | ❌ 未开始 | 需要 LJSpeech 数据集 |
| Section 6 实验运行 | ❌ 未开始 | 预计 ~14 小时 CPU |
| Section 6 结果推送 GitHub | ❌ 未开始 | - |

---

## 2. 计划方案（已批准）

### 文件结构

```
D:\C\Python\backdoor\audiosealexp\section6_adversarial\
├── config.py               # 共享配置（攻击参数、alpha sweep、样本数）
├── utils.py                # 音频加载/水印/检测工具（返回 tensor 以支持梯度）
├── attacks.py              # 三种攻击算法实现
├── training.py             # 代理检测器 & 黑盒分类器训练
├── evaluation.py           # 评价指标 & Figure 7 风格绘图
├── run_whitebox.py         # 白盒攻击入口
├── run_semi_blackbox.py    # 半黑盒攻击入口
├── run_blackbox.py         # 黑盒攻击入口
├── run_all.py              # 运行全部
└── results/                # 输出目录
```

### 三种攻击概述

| 攻击类型 | 描述 | 是否需要训练 | 预计运行时间 |
|----------|------|-------------|-------------|
| 白盒 | 知道检测器架构和参数，用 Adam 优化对抗扰动 | ❌ 不需要 | ~4 小时（80样本×5 alpha） |
| 半黑盒 | 训练同架构代理检测器，用代理做白盒攻击 | ✅ 需要训练代理检测器 | ~6 小时（含训练） |
| 黑盒 | 训练 1D CNN 分类器，用分类器做攻击目标 | ✅ 需要训练分类器 | ~4 小时（含训练） |

### 攻击算法（论文 Appendix D.5）

```
输入: 水印音频 x, 检测器 D
初始化: adv = zeros_like(x)
for step in range(100):  # Adam 100 步
    perturbed = x + sigma * tanh(adv)   # sigma = 1e-3
    loss = -D(perturbed)[:, 1, :].mean()  # 最小化检测分数（channel 1 = P(watermark)）
    adv = adam_step(adv, grad(loss))
输出: x + sigma * tanh(adv)
```

### 关键参数

```python
@dataclass
class Config:
    # 攻击参数（论文 Appendix D.5）
    num_steps: int = 100           # Adam 优化步数
    sigma: float = 1e-3            # 扰动幅度
    lr: float = 0.1                # Adam 学习率

    # alpha sweep（复现 Figure 7 曲线，不同攻击强度）
    alpha_values: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0]

    # 评价指标
    # 检测错误率 = (FNR + FPR) / 2
    # 音频质量 = SI-SNR（替代论文的 PESQ）

    # CPU 可行样本数
    whitebox_samples: int = 80
    semi_bb_samples: int = 80
    blackbox_samples: int = 60

    # 数据集
    data_root: str = r"D:\C\Python\backdoor\LJSpeech-1.1"
    sample_rate: int = 16000
    max_duration: float = 10.0
    watermark_bits: int = 16
    device: str = "cpu"
```

### 已知问题

1. **fast_1.25x 攻击实现有差异**：我们用 `scipy.signal.resample` + 零填充，严重破坏水印；论文可能用相位声码器
2. **PESQ 库安装失败**：缺少 MSVC 编译工具，用 SI-SNR 替代
3. **半黑盒/黑盒需要训练代理模型**：无预训练模型可用，需从头训练
4. **CPU 运行较慢**：白盒攻击每样本约 36 秒，总需 ~14 小时

### 依赖

- 已安装：`audioseal 0.2.0`, `torch`, `numpy`, `scipy`, `soundfile`, `pystoi`, `matplotlib`, `scikit-learn`
- 数据集：LJSpeech-1.1（已有，`D:\C\Python\backdoor\LJSpeech-1.1\`）
- 不需要 GPU

---

## 3. 恢复步骤

### 第一步：创建目录和基础文件

```bash
mkdir -p D:\C\Python\backdoor\audiosealexp\section6_adversarial\results
```

创建 `config.py`, `utils.py`, `attacks.py`, `training.py`, `evaluation.py`。

### 第二步：白盒攻击（无需训练）

```bash
cd D:\C\Python\backdoor\audiosealexp\section6_adversarial
python run_whitebox.py
```

白盒攻击直接使用 AudioSeal 官方检测器，不需要训练任何模型。先用 5 个样本做 smoke test 验证。

### 第三步：半黑盒攻击（需训练代理检测器）

```bash
python run_semi_blackbox.py
```

使用 `audioseal.builder.create_detector()` 创建同架构模型，在 LJSpeech 上训练代理检测器。

### 第四步：黑盒攻击（需训练分类器）

```bash
python run_blackbox.py
```

训练简单 1D CNN 分类器，用分类器做攻击目标。

### 第五步：运行全部（可选）

```bash
python run_all.py
```

### 第六步：推送结果到 GitHub

将结果文件复制到 `paper/audio/AudioSeal/experiment/` 并推送。

---

## 4. 参考文件

| 文件 | 路径 | 用途 |
|------|------|------|
| 计划文件 | `C:\Users\ddd\.claude\plans\hazy-puzzling-starlight.md` | 详细实现计划 |
| Table 3 实验脚本 | `D:\C\Python\backdoor\audiosealexp\audioseal_table3_experiment.py` | 复用模式（音频加载/水印/检测） |
| Table 3 实验报告 | `D:\C\Python\backdoor\audiosealexp\table3_experiment_report.md` | Table 3 结果参考 |
| GitHub 仓库 | `C:\Users\ddd\Desktop\论文分类\watermarking-learning-notebook\` | 推送目标 |

---

## 5. Figure 7 风格输出示例

实验完成后应生成类似论文 Figure 7 的曲线图：

```
检测错误率 (y 轴) vs PESQ/SI-SNR (x 轴)
- 白盒：左下角（低质量、低错误率 → 最有效）
- 半黑盒：中间
- 黑盒：右上角（高质量、高错误率 → 最弱）
- 每种攻击 5 个点（alpha = 0.25, 0.5, 1.0, 2.0, 4.0）
```

---

*创建时间: 2026-04-13*
*待继续: 实现代码 → 训练代理模型 → 运行实验 → 推送 GitHub*
