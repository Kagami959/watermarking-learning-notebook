# AudioSeal Section 6 — 黑白盒对抗攻击训练思路

## 背景

AudioSeal 论文 Section 6 研究的是 **对抗性水印移除（Adversarial Watermark Removal）**：攻击者能否通过对水印音频施加微小扰动，使检测器无法正确检测水印？

攻击者面临的约束是：对检测器的了解程度不同，由此衍生三种攻击范式——白盒、半黑盒、黑盒。三者共享同一个核心优化算法，区别仅在于 **优化目标的来源**。

---

## 共同的攻击算法骨架

三种攻击的优化过程完全一致（论文 Appendix D.5）：

```
输入: 水印音频 x, 攻击目标模型 M, 攻击强度 alpha
初始化: adv = zeros_like(x)
sigma = 1e-3 * alpha

for step in range(100):                    # Adam 优化 100 步
    perturbed = x + sigma * tanh(adv)      # tanh 保证扰动有界
    loss = -M(perturbed)[:, 1, :].mean()   # 最小化 P(watermark)
    adv = adam_step(adv, grad(loss))        # 梯度回传，更新扰动

输出: (x + sigma * tanh(adv)).clamp(-1, 1)
```

关键设计：
- **`sigma * tanh(adv)`**：用 tanh 将无界的优化变量 adv 映射到 `[-sigma, sigma]` 范围内，控制扰动幅度
- **负号取反**：优化目标是最小化检测分数（即让检测器认为"没有水印"）
- **alpha 参数**：sweep `[0.25, 0.5, 1.0, 2.0, 4.0]` 生成不同攻击强度的曲线，对应论文 Figure 7

---

## 三种攻击的训练思路

### 1. 白盒攻击（White-box）—— 无需训练

**攻击者假设**：完全知道检测器的架构和权重参数。

**思路**：
1. 直接使用 AudioSeal 官方预训练检测器 `audioseal_detector_16bits` 作为攻击目标
2. 对水印音频执行上述 Adam 优化，梯度直接从真实检测器反传
3. 因为攻击者完全了解目标模型，生成的对抗扰动 **精确针对** 该检测器

**训练开销**：无。攻击目标就是现成的预训练模型。

**预期效果**：最强攻击。扰动完全拟合检测器的决策边界，检测错误率最高，但音频质量（SI-SNR）也最差。

**代码流程**：
```
load_audio → watermark_audio → whitebox_attack(audio, real_detector) → detect(real_detector)
```

---

### 2. 半黑盒攻击（Semi Black-box）—— 训练代理检测器

**攻击者假设**：知道检测器的架构（网络结构），但 **不知道权重参数**。

**思路**（迁移攻击 / Transfer Attack）：
1. 用相同架构创建一个 **代理检测器（Surrogate Detector）**
2. 在本地数据上训练代理检测器，使其学会"检测水印"
3. 对代理检测器做白盒攻击（梯度从代理反传）
4. 生成的对抗样本 **迁移到真实检测器** 上测试效果

**代理检测器训练细节**：

| 项目 | 设置 |
|------|------|
| 架构 | 与 AudioSeal 官方检测器完全相同（`audioseal_detector_16bits`） |
| 权重初始化 | 加载预训练权重后 **重新初始化**（`reset_parameters()`），避免直接复制 |
| 训练数据 | LJSpeech 原始音频 + 用预训练生成器 on-the-fly 生成水印版本 |
| 标签 | 水印样本 label=1，非水印样本 label=0，随机 50/50 采样 |
| 损失函数 | `BCEWithLogitsLoss`，取 channel 1（P(watermark)）的时域均值作为预测 |
| 优化器 | Adam, lr=1e-3, 20 epochs, batch_size=8 |

**为什么需要重新初始化权重？**
如果直接用预训练权重，代理检测器就等于真实检测器，半黑盒退化为白盒。重新初始化后，代理检测器只能通过本地数据重新学习水印特征，模拟攻击者"自己训练了一个同架构模型"的场景。

**迁移性原理**：
不同模型虽然权重不同，但学到的特征空间往往有重叠。在代理检测器上有效的对抗扰动，大概率在真实检测器上也部分有效——这就是对抗样本的 **迁移性（transferability）**。

**代码流程**：
```
train_surrogate_detector()  →  得到代理检测器
load_audio → watermark_audio → semi_blackbox_attack(audio, surrogate, real_detector) → detect(real_detector)
```

核心区别：优化时梯度从 **代理检测器** 反传，但最终评估用的是 **真实检测器**。

---

### 3. 黑盒攻击（Black-box）—— 训练 1D CNN 分类器

**攻击者假设**：既不知道检测器架构，也不知道权重。只能采集 (音频, 是否水印) 的二元标签。

**思路**：
1. 训练一个与 AudioSeal 检测器 **架构完全不同** 的简单分类器
2. 用该分类器作为攻击目标做白盒优化
3. 生成的对抗样本迁移到真实检测器上

**黑盒分类器架构（自定义 1D CNN）**：

```
输入: [B, 1, T]  音频波形
  ↓
Conv1d(1→32, k=80, s=4) + BN + ReLU     # 提取低级特征
  ↓
Conv1d(32→64, k=40, s=4) + BN + ReLU    # 中级特征
  ↓
Conv1d(64→128, k=20, s=4) + BN + ReLU   # 高级特征
  ↓
AdaptiveAvgPool1d(1)                     # 全局平均池化
  ↓
Linear(128→1) + Sigmoid                  # 输出 P(watermark) ∈ [0,1]
```

与 AudioSeal 检测器（编码器-解码器结构，输出逐样本点的水印概率）完全不同——这个分类器是一个简单的"黑箱"，只输出整段音频的水印概率标量。

**分类器训练细节**：

| 项目 | 设置 |
|------|------|
| 架构 | 自定义 3 层 1D CNN（与 AudioSeal 完全不同） |
| 训练数据 | 同样用 LJSpeech + on-the-fly 水印生成 |
| 损失函数 | `BCELoss`（因为模型输出已过 Sigmoid） |
| 优化器 | Adam, lr=1e-3, 30 epochs, batch_size=8 |
| 评估指标 | 二分类准确率 |

**为什么黑盒攻击最弱？**
- 分类器架构与真实检测器完全不同，特征空间差异大
- 分类器输出的是全局概率（一个标量），而检测器输出逐样本点的时域概率（一个序列）
- 迁移性更差：为简单分类器找到的对抗方向，在复杂检测器上可能完全无效

**代码流程**：
```
train_blackbox_classifier()  →  得到 1D CNN 分类器
load_audio → watermark_audio → blackbox_attack(audio, classifier) → detect(real_detector)
```

---

## 三种攻击的对比总结

| 维度 | 白盒 | 半黑盒 | 黑盒 |
|------|------|--------|------|
| 攻击者知识 | 架构 + 权重 | 仅架构 | 啥都不知道 |
| 是否需要训练 | 否 | 是（代理检测器） | 是（1D CNN 分类器） |
| 训练目标模型 | 无（直接用官方） | 同架构重新训练 | 自定义简单 CNN |
| 优化时梯度来源 | 真实检测器 | 代理检测器 | 1D CNN 分类器 |
| 评估目标 | 真实检测器 | 真实检测器（迁移） | 真实检测器（迁移） |
| 攻击效果 | 最强 | 中等 | 最弱 |
| 音频质量代价 | 最高（扰动大） | 中等 | 最低（扰动小） |

---

## Alpha 参数的作用

Alpha 是攻击强度的倍率，`sigma = 1e-3 * alpha`：

- **alpha = 0.25**：扰动极小，音频质量好（SI-SNR 高），但水印检测错误率低（攻击弱）
- **alpha = 4.0**：扰动很大，音频质量差（SI-SNR 低），但水印检测错误率高（攻击强）

Sweep 5 个 alpha 值绘制 Figure 7 风格曲线：x 轴为 SI-SNR（音频质量），y 轴为检测错误率。理想情况下：
- 白盒曲线在左下角（低质量 + 低错误率 → 最有效移除水印）
- 黑盒曲线在右上角（高质量 + 高错误率 → 攻击效果最弱，但保真度最好）

---

## 评价指标

**检测错误率（Detection Error Rate）** = (FNR + FPR) / 2
- FNR（False Negative Rate）：水印样本被判定为无水印的比例（攻击成功的标志）
- FPR（False Positive Rate）：无水印样本被误判为有水印的比例
- 通过遍历阈值 `[0.01, 1.00]` 找到使错误率最小的最优阈值

**音频质量 SI-SNR（Scale-Invariant Signal-to-Noise Ratio）**
- 衡量对抗扰动对原始音频的破坏程度
- 论文原用 PESQ，因 PESQ 库需要 MSVC 编译工具，用 SI-SNR 替代

---

## 完整实验流程

```
白盒:  load → watermark → whitebox_attack(x, detector) → 评估 detector
                                                                 ↓
半黑盒: load → watermark → train_surrogate → semi_bb_attack(x, surrogate) → 评估 detector
                                                                 ↓
黑盒:  load → watermark → train_classifier → bb_attack(x, classifier) → 评估 detector
```

每种攻击 × 5 个 alpha 值 → 绘制 Detection Error Rate vs SI-SNR 曲线 → 与论文 Figure 7 定性对比。

---

*基于 AudioSeal 论文 Section 6 及 `D:\C\Python\backdoor\audiosealexp\section6_adversarial\` 实验代码整理*
