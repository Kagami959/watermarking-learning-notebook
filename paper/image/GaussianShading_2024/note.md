# Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models
- **作者/年份/venue**: Zijin Yang, Kai Zeng, Kejiang Chen, Han Fang, Weiming Zhang, Nenghai Yu / 2024 / CVPR 2024 (USTC & NUS)
- **核心问题**: 现有扩散模型水印方法要么损害模型性能（后处理方法），要么需要额外训练（微调方法），要么限制采样随机性（Tree-Ring）。能否在不损失模型性能的前提下嵌入水印？
- **方法分类**: 图像水印 / 扩散模型 / 固有式水印
- **核心创新**: 提出**性能无损**且**无需训练**的水印方案。将水印映射到服从标准高斯分布的潜在表示上，与无水印扩散模型的潜在表示在统计上不可区分，从而实现性能无损（有理论证明）。
- **关键技术细节**:
  - **整体流程**（见论文Figure 3）：
    ```
    水印嵌入：
    1. 将k-bit二进制水印 s 用流密码(ChaCha20)加密 → 随机化水印 m（均匀分布）
    2. 将 m 的每l-bit视为整数 y ∈ [0, 2^l - 1]
    3. 将标准高斯分布 N(0,I) 的概率密度等分为 2^l 个累积概率区间
    4. 当 y=i 时，采样 z^s_T 落入第i个区间 → 驱动分布保持采样(Distribution-Preserving Sampling)
    5. 对 z^s_T 做去噪生成带水印图像 X^s

    水印提取：
    1. 用编码器 E 将图像 X'^s 映射回潜在空间 z'^s_0 = E(X'^s)
    2. DDIM Inversion 估计加性噪声 → z'^s_T ≈ z^s_T
    3. 逆变换 z'^s_T → 比特流 m'
    4. 用密钥 K 解密 m' → s'_d
    5. 投票机制恢复原始水印 s（每个水印有 f_c·f_hw^2 份副本）
    ```
  - **性能无损证明**: 水印后的潜在表示 z^s_T 服从标准高斯分布 N(0,I)，与无水印模型的采样分布在统计上不可区分。任何多项式时间的测试者都无法区分水印图像和正常生成图像。
  - **与Tree-Ring的关键区别**:
    - Tree-Ring 在噪声空间嵌入固定环形模式 → **限制了采样随机性**，影响生成质量
    - Gaussian Shading 通过分布保持采样 → 水印噪声**仍服从标准高斯分布**，不改变采样分布
  - **与Zodiac的关系**: Zodiac基于Tree-Ring改进，对已有图片先做DDIM反演再嵌入水印；Gaussian Shading则在生成阶段直接嵌入，思路更简洁
  - **容量**: 实际容量 k = l × c × h × w / (f_c · f_hw^2) bits，默认配置256 bits
- **实验设置**:
  - 模型：Stable Diffusion V1.4/V2.0/V2.1
  - 基线：DwtDct, DwtDctSvd, RivaGAN, Stable Signature, Tree-Ring
  - 攻击：JPEG压缩、随机裁剪、随机丢弃、高斯模糊、中值滤波、高斯噪声、椒盐噪声、缩放、亮度调整
  - 评价指标：TPR@FPR=10^-6、Bit Accuracy、FID、CLIP-Score
  - 硬件：单卡RTX 3090
- **实验结果亮点**:
  - FID t-test p值0.3567（>0.05），证明性能无损
  - 干净图像TPR=1.000，对抗攻击后TPR=0.997（远超Tree-Ring的0.894和Stable Signature的0.502）
  - Bit Accuracy: 干净0.9999，对抗攻击后0.9753（远超所有基线）
  - 与不同采样方法(DDIM/UniPC/PNDM/DEIS/DPMSolver)兼容
- **优点/局限**:
  - 优点：
    - **性能无损**：有严格理论证明，FID无显著变化
    - **无需训练**：即插即用，不修改模型参数
    - **鲁棒性极强**：水印扩散到整个语义空间，对抗攻击后仍保持高准确率
    - **支持多比特**：可同时用于版权保护（检测）和来源追溯（归属）
    - 与多种采样器和推理/反演步数兼容
  - 局限：
    - 依赖DDIM Inversion的精度（但实验表明即使10步反演也能保持高准确率）
    - 需要流密码密钥K的安全管理
    - 容量与鲁棒性存在trade-off（f_hw越大容量越小，但鲁棒性越强）
- **我的思考/疑问**:
  - **PPT学习笔记**：PPT中将Gaussian Shading与Tree-Ring对比——两者都是在扩散模型生成过程中嵌入水印，但Gaussian Shading的理论基础更扎实（分布保持 vs 固定模式）
  - "将水印伪装成标准高斯噪声"的核心思想非常优雅：既然扩散模型的初始噪声就是高斯分布，那让水印后的噪声也服从同样的分布，自然就"无损"了
  - 流密码加密步骤确保了水印的随机性，同时提供了安全保证
  - 投票机制（多个副本取多数）是提升鲁棒性的关键工程设计
  - 该方法与PAI的思路对比：PAI通过偏转去噪轨迹嵌入水印（语义耦合更强），Gaussian Shading通过噪声空间的分布保持嵌入水印（理论保证更强）
- **关联文献**: [[TreeRing_Watermarks2023]], [[PAI]], [[StableSignature2023]], [[DDPM]], [[LDM]]
- **代码/数据链接**: https://github.com/zijin-yang/Gaussian-Shading
- **复现笔记**: （后续补充：环境、踩坑、调参、结果截图）
