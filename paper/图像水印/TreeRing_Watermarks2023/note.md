# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust
- **作者/年份/ venue**: Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein / 2023 / NeurIPS 2023
- **核心问题**: 如何为扩散模型生成的图像嵌入鲁棒且不可见的水印？现有方法需要对图像进行后处理修改。
- **方法分类**: 图像水印 / 扩散模型
- **核心创新**: 不在图像生成后修改，而是在初始噪声向量中嵌入环形模式（ring pattern），使水印在扩散过程中自然融入。在傅里叶空间中设计模式，使水印对卷积、裁剪、膨胀、翻转、旋转等操作具有不变性。
- **关键技术细节**:
  - 水印嵌入：
    1. 在初始噪声向量的傅里叶空间中嵌入环形模式
    2. 扩散过程将噪声中的模式自然传播到最终图像
  - 水印检测：
    1. 通过反转扩散过程（DDIM inversion）恢复初始噪声向量
    2. 检查噪声向量中是否存在嵌入的环形模式
  - 鲁棒性来源：傅里叶空间中的环形模式对空间变换（卷积、裁剪、翻转、旋转）具有不变性
- **实验设置**:
  - 模型：Stable Diffusion（作为插件使用）
  - 评价指标：FID、水印检测率、鲁棒性（面对各种攻击）
  - 对比：与频域水印等其他方法对比
- **优点/局限**:
  - 优点：
    - 作为插件可直接应用于任意扩散模型，无需额外训练
    - 对FID影响极小（negligible loss）
    - 对多种空间变换具有鲁棒性
    - 水印在语义上隐藏在图像空间中
  - 局限：（待补充）
- **我的思考/疑问**:
  - 与PAI对比：PAI在provider端操作，Tree-Ring在噪声空间操作
  - 与Stable Signature对比：Tree-Ring不需要微调模型，而Stable Signature需要微调解码器
  - 傅里叶空间的环形模式设计非常巧妙——利用了卷积在傅里叶空间变为乘法的性质
  - DDIM inversion是检测的关键——需要可靠的噪声恢复能力
- **关联文献**: [[PAI]], [[StableSignature2023]], [[DDPM]], [[LDM]]
- **代码/数据链接**: https://github.com/YuxinWenRick/tree-ring-watermark
- **复现笔记**: （后续补充：环境、踩坑、调参、结果截图）
