# 音频生成式水印新论文整理

> 与 Tree-Ring / ZODiAC 思路类似的音频水印论文 —— 利用生成过程嵌入水印
> 整理日期：2026-05-12

---

## 目录

1. [GROOT — ACM MM 2024](#1-groot-acm-mm-2024)
2. [Smark — 2025](#2-smark-2025)
3. [TriniMark — 2025](#3-trinimark-2025)
4. [SOLIDO — 2025](#4-solido-2025)
5. [TraceableSpeech — Interspeech 2024](#5-traceablespeech-interspeech-2024)
6. [Latent Watermarking of Audio Generative Models — 2024](#6-latent-watermarking-of-audio-generative-models-2024)
7. [对比总结](#7-对比总结)

---

## 1. GROOT — ACM MM 2024

**标题：** GROOT: Generating Robust Watermark for Diffusion-Model-Based Audio Synthesis
**作者：** Weizhi Liu, Yue Li, Dongdong Lin, Hui Tian, Haizhou Li（华大/港中大深圳）
**来源：** arXiv:2407.10471 | ACM MM 2024 (CCF-A)
**文件：** `GROOT_2024.pdf`

### 核心内容

GROOT 是**第一个**将生成式水印引入**音频扩散模型**的工作。它提出了一种新范式：水印生成和音频合成**同时进行**，而不是先生成音频再加后处理水印。

### 方法细节

- **架构**：专用编码器(Encoder)将二进制水印转换为扩散模型可识别的潜变量 → 与高斯潜噪声一起送入参数固定的扩散模型(DiffWave/WaveGrad/PriorGrad) → 轻量化解码器(Decoder)从生成音频中提取水印
- **训练方式**：只训练编码器和解码器，**扩散模型参数冻结**，即插即用
- **损失函数**：联合优化水印嵌入质量与提取准确率

### 创新点

1. **新范式**：首个面向音频扩散模型的生成式水印方案，实现"边生成边嵌入"
2. **即插即用**：扩散模型参数固定，只训练编解码器，可适配任意音频扩散模型
3. **强鲁棒性**：面对复合攻击（多种攻击任意组合）仍保持 ~95% 提取准确率
4. **高容量**：支持高达 5000 bps

### 与 Tree-Ring 的对应关系

| Tree-Ring (图像) | GROOT (音频) |
|:---|:---|
| 在初始噪声傅里叶域嵌入环形水印 | 编码器将水印转为潜变量输入扩散过程 |
| DDIM反演检测水印 | 轻量化解码器从生成音频提取水印 |
| 无需训练，即插即用 | 仅训练编解码器，扩散模型不动 |

---

## 2. Smark — 2025

**标题：** Smark: A Watermark for Text-to-Speech Diffusion Models via Discrete Wavelet Transform
**作者：** Yichuan Zhang, Chengxin Li, Yujie Gu（九州大学）
**来源：** arXiv:2512.18791
**文件：** `Smark_2025.pdf`

### 核心内容

Smark 是一个**模型无关的通用水印方案**，适用于所有 TTS 扩散模型。核心思路是利用**离散小波变换(DWT)** 在反向扩散过程中将水印嵌入到听觉感知稳定的低频子带(LL)。

### 方法细节

- **通用性来源**：所有 TTS 扩散模型共享相同的反向扩散数学范式（马尔可夫链去噪），Smark 在这一层操作，不依赖特定模型架构
- **DWT嵌入**：
  - 对反向扩散过程中某时间步的梅尔频谱图做 DWT → 分解为 LL(低频近似)、LH、HL、HH 四个子带
  - 水印只嵌入 **LL 子带**（携带核心结构信息，抗干扰强）
  - IDWT 重构梅尔频谱图 → 继续反向扩散
- **水印编解码器**：轻量级神经网络（全连接层 + 膨胀卷积 + 强度预测器）
- **验证框架**：基于二项分布的假设检验，控制 FPR/FNR

### 创新点

1. **模型无关**：首次实现适用于**所有** TTS 扩散模型的通用水印方案
2. **DWT低频嵌入**：利用 DWT 多分辨率分析能力，将水印嵌入感知稳定区域，兼顾不可感知性和鲁棒性
3. **自适应强度**：动态预测嵌入强度，在感知掩蔽区域加强嵌入
4. **统计验证**：假设检验框架确保统计可靠性

### 与 Tree-Ring 的对应关系

| Tree-Ring | Smark |
|:---|:---|
| 傅里叶域环形水印 | DWT域低频子带水印 |
| 初始噪声嵌入 | 反向扩散过程中间时间步嵌入 |
| 图像扩散模型 | TTS音频扩散模型 |

---

## 3. TriniMark — 2025

**标题：** TriniMark: A Robust Generative Speech Watermarking Method for Trinity-Level Traceability
**作者：** Yue Li, Weizhi Liu, Kaiqing Lin, Dongdong Lin, Kassem Kallas（华大/华师大/深大）
**来源：** arXiv:2504.20532
**文件：** `TriniMark_2025.pdf`

### 核心内容

TriniMark 是面向扩散模型的生成式语音水印框架，实现**三联溯源**：
- **内容级** (Content-level)：水印信息可追溯
- **模型级** (Model-level)：可识别来源模型
- **用户级** (User-level)：可识别生成请求的具体用户

### 方法细节

- **两阶段训练**：
  - **Stage I**：预训练时域水印编码器-解码器（带噪声层增强鲁棒性）
  - **Stage II**：波形引导微调(WGFT)，将水印能力注入扩散模型
- **可变水印训练**：训练时使用不同水印消息，推理时可分配不同水印给不同用户
- **架构**：轻量级时域编码器 + 时域门控卷积解码器(Temporal-aware Gated Convolutional Decoder)
- **容量**：500 bps

### 创新点

1. **三联溯源**：首个同时支持内容、模型、用户三级溯源的语音水印方法
2. **可扩展用户追踪**：可变水印训练策略 → 推理时不同用户获得不同水印 → 大规模部署可行
3. **波形引导微调**：将预训练编解码器的水印能力有效迁移到扩散模型
4. **高质量-鲁棒性-容量权衡**：感知质量无损、高提取准确率、500bps

### 与 Tree-Ring/ZODiAC 的对应关系

| ZODiAC | TriniMark |
|:---|:---|
| 对已有图像做DDIM反演→嵌入→再生成 | 两阶段训练将水印注入扩散生成过程 |
| 只支持所有权验证（0-bit） | 多bit信息，支持用户级溯源 |
| 图像领域 | 语音扩散模型 |

---

## 4. SOLIDO — 2025

**标题：** SOLIDO: A Robust Watermarking Method for Speech Synthesis via Low-Rank Adaptation
**作者：** Yue Li, Weizhi Liu, Dongdong Lin（华大）
**来源：** arXiv:2504.15035
**文件：** `SOLIDO_2025.pdf`

### 核心内容

SOLIDO 首次将 **LoRA（低秩适配）参数高效微调**引入语音生成式水印，大幅降低训练开销的同时保持高鲁棒性。

### 方法细节

- **LoRA 微调**：冻结扩散模型原始参数，只训练两个低秩矩阵 A 和 B（r << d,k）
- **水印编码器**：仅三个操作（全连接+ReLU），将水印转换为扩散模型输入的潜变量
- **水印解码器**：基于**深度可分离卷积**，可直接处理**变长输入**（无需截断或填充）
- **语音驱动轻量微调策略(SDFT)**：联合优化语音质量和提取准确率
- **容量**：2000 bps

### 创新点

1. **LoRA水印**：首个将参数高效微调(PEFT)用于语音水印的工作，训练参数量大幅减少
2. **变长输入处理**：深度可分离卷积解码器直接处理任意长度音频，对裁剪/时间拉伸攻击鲁棒
3. **极高提取率**：单一攻击 99.20%、复合攻击 98.43%
4. **抗时间拉伸**：比 SOTA 方法提升 ~23%

### 计算开销对比

| 方法 | 训练参数量 | 训练成本 |
|:---|:---:|:---:|
| 全参数微调(FPT) | 100% | 高 |
| 参数冻结(PFT) | 仅编解码器 | 中 |
| **SOLIDO (LoRA)** | **极少量低秩矩阵** | **低** |

---

## 5. TraceableSpeech — Interspeech 2024

**标题：** TraceableSpeech: Towards Proactively Traceable Text-to-Speech with Watermarking
**作者：** Junzuo Zhou, Jiangyan Yi, Tao Wang, Jianhua Tao, Ye Bai et al.（中科院自动化所/清华）
**来源：** arXiv:2406.04840 | Interspeech 2024
**文件：** `TraceableSpeech_2024.pdf`

### 核心内容

TraceableSpeech 是一个**端到端可溯源TTS模型**，在语音合成过程中直接生成含水印语音，而不是合成后再添加水印。基于 VALL-E 架构（神经编解码+语言模型）。

### 方法细节

- **两阶段架构**：
  - Stage 1：神经编解码器(HiFiCodec) + 水印机制联合训练
  - Stage 2：语言模型(VALL-E) + 水印机制联合训练
- **逐帧广播嵌入(Frame-wise Imprinting)**：水印编码为向量后在时间轴上广播，与每一帧的语音潜特征融合
- **水印提取器**：ResNet 提取 r-vector → 多层线性层预测每位数字
- **攻击模拟训练**：7种攻击（重采样、噪声、丢包等），加权采样

### 创新点

1. **端到端水印TTS**：首次实现 TTS 模型直接生成含水印语音，取代后处理范式
2. **帧级水印**：逐帧广播嵌入 → 任意时长语音都支持，且抗拼接攻击
3. **高鲁棒性**：0.3秒超短语音中嵌入4位64进制水印，提取准确率 >95%
4. **时间灵活性**：不像 WavMark 需要固定长度分段

### 与 ZODiAC 的对应关系

| ZODiAC | TraceableSpeech |
|:---|:---|
| 已有图像→DDIM反演→嵌入→再生成 | 语音在编解码器潜空间嵌入水印 |
| 迭代优化水印质量 | 端到端训练水印编解码器 |
| 基于 Stable Diffusion | 基于 VALL-E (LM + Codec) |

---

## 6. Latent Watermarking of Audio Generative Models — 2024

**标题：** Latent Watermarking of Audio Generative Models
**作者：** Robin San Roman, Pierre Fernandez, Antoine Deleforge, Yossi Adi, Romain Serizel（Meta FAIR / Inria / 希伯来大学）
**来源：** arXiv:2409.02915
**文件：** `LatentWatermarking_Audio_2024.pdf`

### 核心内容

该方法不直接在推理时嵌入水印，而是在**训练数据**中嵌入水印，使训练出的潜空间生成模型（以 MusicGen 为例）天然产出含水印音频。即使模型被开源、微调，输出仍然可检测。

### 方法细节

- **三步流程**：
  1. 训练水印生成器/检测器（基于 AudioSeal），增强对 EnCodec 压缩的鲁棒性
  2. 对训练集中的音频打上水印 → 用 EnCodec 分词 → 训练 MusicGen (Transformer LM)
  3. 推理时，MusicGen 生成的音频天然携带水印，可用检测器识别
- **核心原理**：只要水印能抵抗 EnCodec 的压缩/量化，训练后的 LM 学到的 token 分布就天然含水印

### 创新点

1. **数据级水印注入**：不同于推理时嵌入，水印在训练数据层面注入，模型输出天然含水印
2. **开源模型保护**：即使模型被完全开源，攻击者也无法绕过水印
3. **微调鲁棒**：即使在水印数据集上微调，检测率仍 >75% (@10⁻³ FPR)
4. **解码器无关**：无论使用什么解码器/vocoder，输出都可检测

### 与 Tree-Ring 的对比

| Tree-Ring | Latent Watermarking |
|:---|:---|
| 推理时在初始噪声嵌入 | 训练时在训练数据嵌入 |
| 需要 DDIM 反演检测 | 用 AudioSeal 检测器直接检测 |
| 即插即用，无需训练 | 需要重新训练生成模型 |
| 适合保护生成过程 | 适合保护开源模型 |

---

## 7. 对比总结

### 按与 Tree-Ring/ZODiAC 的相似度排序

| 论文 | 相似度 | 生成过程嵌入 | 训练需求 | 载体模型 | 容量 | ROBOUST |
|:-----|:------:|:----------:|:--------:|:--------:|:---:|:-------:|
| **GROOT** | ⭐⭐⭐⭐ | ✅ 编解码器注入 | 仅编解码器 | DiffWave等 | 5000bps | 95%复合攻击 |
| **Smark** | ⭐⭐⭐ | ✅ DWT+反向扩散 | 编解码器+TTS | 任意TTS扩散模型 | 可变 | 强 |
| **TriniMark** | ⭐⭐⭐ | ✅ 两阶段微调 | 编解码器+WGFT | 语音扩散模型 | 500bps | 三联溯源 |
| **SOLIDO** | ⭐⭐⭐ | ✅ LoRA微调 | LoRA+编解码器 | 语音扩散模型 | 2000bps | 99.2%/98.4% |
| **TraceableSpeech** | ⭐⭐⭐ | ✅ 逐帧广播嵌入 | 端到端训练 | VALL-E (Codec+LM) | 4位64进制 | 抗拼接攻击 |
| **Latent Watermarking** | ⭐⭐ | ✅ 训练数据注入 | 重新训练LM | MusicGen | 0-bit检测 | 75%@10⁻³FPR |

### 按应用场景选择

- **想快速适配现有音频扩散模型** → **GROOT**（即插即用，只需训练编解码器）
- **需要通用方案（不同TTS模型）** → **Smark**（模型无关，DWT低频嵌入）
- **需要用户级溯源（大规模部署）** → **TriniMark**（三联溯源，可变水印）
- **计算资源有限** → **SOLIDO**（LoRA微调，参数极少）
- **基于Codec/TTS的场景** → **TraceableSpeech**（VALL-E架构，端到端）
- **保护开源模型权重** → **Latent Watermarking**（数据级注入，微调仍可检测）
