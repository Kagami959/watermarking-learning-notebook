# High-Resolution Image Synthesis with Latent Diffusion Models (LDM)
- **作者/年份/ venue**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer / 2022 / CVPR 2022
- **核心问题**: 扩散模型直接在像素空间操作，训练消耗数百GPU天，推理昂贵。如何在保留质量的同时大幅降低计算成本？
- **方法分类**: 生成式AI基础 / 扩散模型
- **核心创新**: 将扩散过程从像素空间转移到预训练自编码器的latent空间，首次在复杂度削减和细节保留之间达到近最优平衡。引入交叉注意力层实现文本/边界框等条件控制。
- **关键技术细节**:
  - 两阶段训练：
    1. 训练感知压缩自编码器（perceptual compression autoencoder），将图像压缩到latent空间
    2. 在latent空间中训练扩散模型
  - 条件机制：通过交叉注意力（cross-attention）层将文本、语义图等条件注入UNet
  - 关键优势：latent空间的扩散模型可以用卷积方式实现高分辨率合成
- **实验设置**:
  - 数据集：ImageNet, CelebA-HQ, LSUN, LAION
  - 评价指标：FID, IS, Precision/Recall
  - 任务：图像修复、类条件合成、文本到图像、无条件生成、超分辨率
  - 对比：与pixel-based DM相比，大幅降低计算需求，同时达到SOTA
- **优点/局限**:
  - 优点：
    - 计算效率显著提升（相比pixel-based DM）
    - 灵活的条件控制机制（cross-attention）
    - 成为Stable Diffusion的基础架构
    - 在多个任务上达到SOTA
  - 局限：依赖预训练自编码器的质量
- **我的思考/疑问**:
  - LDM是理解PAI的关键前置知识——PAI利用了扩散模型在latent空间中去噪的特性
  - latent空间的压缩率为f=4时效果最好，f=8和f=16会导致细节损失
  - PAI的即插即用设计正是利用了LDM的反向去噪过程，在latent空间中嵌入水印
- **关联文献**: [[DDPM]], [[PAI]], [[TreeRing_Watermarks2023]], [[StableSignature2023]]
- **代码/数据链接**: https://github.com/CompVis/latent-diffusion
- **复现笔记**: （后续补充：环境、踩坑、调参、结果截图）
