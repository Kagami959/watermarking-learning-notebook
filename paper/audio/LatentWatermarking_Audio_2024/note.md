# Latent Watermarking of Audio Generative Models
- **作者/年份/venue**: Robin San Roman, Pierre Fernandez, Antoine Deleforge, Yossi Adi, Romain Serizel (Meta FAIR / Inria / 希伯来大学) / 2024 / arXiv:2409.02915
- **核心问题**: 现有生成式水印在推理时嵌入，但一旦模型被开源或微调，水印机制可能被绕过。能否在模型层面永久嵌入水印，使其无法被移除？
- **方法分类**: 音频水印 / 生成模型 / 数据级水印
- **核心创新**: 不直接在推理时嵌入水印，而是在**训练数据**中嵌入水印，使训练出的潜空间生成模型（以MusicGen为例）天然产出含水印音频。即使模型被开源、微调，输出仍然可检测。
- **关键技术细节**:
  - **三步流程**：
    1. 训练水印生成器/检测器（基于AudioSeal），增强对EnCodec压缩的鲁棒性
    2. 对训练集中的音频打上水印 → 用EnCodec分词 → 训练MusicGen (Transformer LM)
    3. 推理时，MusicGen生成的音频天然携带水印，可用检测器识别
  - **核心原理**：只要水印能抵抗EnCodec的压缩/量化，训练后的LM学到的token分布就天然含水印
- **实验设置**:
  - 模型：MusicGen (Meta的音频生成LM)
  - 基线：推理时嵌入的水印方法
  - 评价指标：检测率、FPR、音频质量
- **实验结果亮点**:
  - 即使在水印数据集上微调，检测率仍>75% (@10⁻³ FPR)
- **优点/局限**:
  - 优点：
    - 数据级水印注入：不同于推理时嵌入，水印在训练数据层面注入
    - 开源模型保护：即使模型被完全开源，攻击者也无法绕过水印
    - 微调鲁棒：微调后检测率仍>75%
    - 解码器无关：无论使用什么解码器/vocoder，输出都可检测
  - 局限：
    - 需要重新训练生成模型（计算成本高）
    - 目前仅支持0-bit检测，不支持多比特信息
    - 水印可能被针对性的对抗训练削弱
- **我的思考/疑问**:
  - 与Tree-Ring的对比很有意思：
    - Tree-Ring推理时在初始噪声嵌入，需要DDIM反演检测
    - Latent Watermarking在训练数据嵌入，用AudioSeal检测器直接检测
    - Tree-Ring即插即用无需训练，Latent Watermarking需要重新训练生成模型
    - Tree-Ring适合保护生成过程，Latent Watermarking适合保护开源模型
  - 这种方法最适合的场景是：公司要开源一个音频生成模型，希望即使被第三方微调后仍能检测模型输出
- **关联文献**: [[AudioSeal]], [[TreeRing_Watermarks2023]], [[GROOT_2024]]
- **代码/数据链接**: arXiv:2409.02915
- **复现笔记**: （后续补充）
