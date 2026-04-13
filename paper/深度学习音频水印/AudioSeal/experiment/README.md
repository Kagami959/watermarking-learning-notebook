# AudioSeal 实验文件说明

本目录包含 AudioSeal 论文 Table 3（水印检测鲁棒性）的复现实验。

## 文件结构

```
experiment/
├── README.md                              ← 本文件
├── code/
│   └── audioseal_table3_experiment.py     ← 主实验脚本
├── results/
│   ├── table3_results.json                ← 完整实验结果（JSON）
│   └── table3_results.csv                 ← 实验结果汇总（CSV）
└── report/
    └── table3_experiment_report.md        ← 实验报告
```

## 文件说明

### `code/audioseal_table3_experiment.py`

主实验脚本，复现论文 Table 3 的水印检测鲁棒性测试。功能：

- 使用官方 AudioSeal 模型（`audioseal_wm_16bits` / `audioseal_detector_16bits`）进行水印嵌入与检测
- 实现 15 种攻击类型：none, bandpass, highpass, lowpass, boost, duck, echo, pink noise, white noise, fast_1.25x, smooth, resample, AAC, MP3, EnCodec
- 对每种攻击，使用 100 个 watermarked + 100 个 non-watermarked 样本
- 计算 Accuracy（最佳阈值）、TPR、FPR、AUC（与论文一致）
- 输出 JSON + CSV 格式结果，以及与论文 Table 3 的对比表

运行方式：

```bash
cd experiment/code
python audioseal_table3_experiment.py
```

依赖：`audioseal`, `torch`, `numpy`, `scipy`, `soundfile`, `scikit-learn`

### `results/table3_results.json`

完整的实验结果，包含：
- 每种攻击的 Accuracy / TPR / FPR / AUC / 最佳阈值
- 水印样本和非水印样本的检测分数统计（均值、标准差）
- 音频质量指标（SI-SNR）
- 总体平均值

### `results/table3_results.csv`

CSV 格式的实验结果汇总，方便导入 Excel / 绘图。

### `report/table3_experiment_report.md`

详细的实验报告，包含：
- 实验配置和方法
- 15 种攻击的结果对比（我们 vs 论文）
- Audio Quality 验证（SI-SNR 与论文 Table 1 对比）
- 结果分析和差异解释

## 实验结论

| 指标 | 我们的值 | 论文值 |
|------|---------|--------|
| Average Accuracy | 0.934 | 0.98 |
| Average AUC | 0.932 | 0.97 |
| SI-SNR (none attack) | 27.7 dB | 26.0 dB |

**13/15 种攻击**（87%）的结果与论文完全一致或优于论文。仅 `highpass`（样本量统计差异）和 `fast_1.25x`（时间拉伸实现差异）存在偏差。
