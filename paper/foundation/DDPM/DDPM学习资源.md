# DDPM 学习资源汇总

## 推荐学习路径（从易到难）

### 第一步：理论理解

- **Lilian Weng — "What are Diffusion Models?"**
  - URL: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  - 公认最好的扩散模型理论入门，从数学推导到伪代码，涵盖变分下界、得分匹配等核心概念

### 第二步：代码逐行讲解（最推荐）

- **HuggingFace — The Annotated Diffusion Model**
  - URL: https://huggingface.co/blog/annotated-diffusion
  - 逐行解读 DDPM 论文 + PyTorch 代码，涵盖噪声调度、前向/逆向过程、UNet、训练循环
  - 理论与代码结合最好

### 第三步：动手实践

#### 入门级

| 仓库 | Stars | 特点 |
|---|---|---|
| [bot66/MNISTDiffusion](https://github.com/bot66/MNISTDiffusion) | 144 | 最简单，MNIST 上从零实现，模型仅 4.55MB，适合理解核心算法 |
| [awjuliani/pytorch-diffusion](https://github.com/awjuliani/pytorch-diffusion) | 185 | 极简实现，配合作者的 "A Simple Guide to Diffusion Models" 系列博客 |
| [acids-ircam/diffusion_models](https://github.com/acids-ircam/diffusion_models) | 721 | Jupyter Notebook 逐步演示，带可视化，适合探索式学习 |

#### 社区热门

| 仓库 | Stars | 特点 |
|---|---|---|
| [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | 10.5k | 社区最热门，代码干净、单文件可读，支持多种噪声调度和采样方法 |
| [w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) | 643 | 紧密对应 Ho et al. 2020 原论文公式，适合逐公式对照学习 |
| [rosinality/denoising-diffusion-pytorch](https://github.com/rosinality/denoising-diffusion-pytorch) | 404 | 知名 PyTorch 贡献者实现，极简且聚焦 |
| [tqch/ddpm-torch](https://github.com/tqch/ddpm-torch) | 233 | 经过充分测试，文档良好，支持多种数据集和调度策略 |

#### 教育向

| 仓库 | Stars | 特点 |
|---|---|---|
| [explainingai-code/DDPM-Pytorch](https://github.com/explainingai-code/DDPM-Pytorch) | 185 | 专为教学设计，配有配套视频教程 |

### 第四步：对照论文 & 官方实现

- [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) — 论文作者 Jonathan Ho 的**官方实现**（5.1k stars），同时包含 JAX 和 PyTorch 版本
- [w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) — 紧密对应原论文公式，适合逐行对照

### 生产级框架

- [huggingface/diffusers](https://github.com/huggingface/diffusers) (33.3k stars)
  - HuggingFace 完整扩散模型库，支持 DDPM、DDIM 等多种变体
  - 优秀的文档、调度器、训练脚本
  - 实际项目中使用

---

## 相关视频教程

- **Amar — "Diffusion Models from Scratch"**
  - URL: https://www.youtube.com/watch?v=a4Yfz2FxXiY
  - 流行的 YouTube 教程，从零构建 DDPM 并逐行讲解

---

## 学习建议

1. **第一次接触**: Lilian Weng 博客 → HuggingFace Annotated Diffusion → bot66/MNISTDiffusion 动手跑
2. **深入理解**: lucidrains 实现 + w86763777 对照论文
3. **实际使用**: huggingface/diffusers
