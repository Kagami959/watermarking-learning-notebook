# A Watermark for Large Language Models
- **作者/年份/ venue**: John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein / 2023 / ICML 2023
- **核心问题**: LLM生成的文本可能被恶意使用（假新闻、学术作弊等），需要一种能在不影响文本质量的前提下嵌入可检测水印的方案。
- **方法分类**: 文本水印 / LLM水印
- **核心创新**: 提出基于Green/Red token list的水印框架——在生成每个token前通过前token哈希确定随机的"绿色token集"，在采样时softly promote绿色token，使水印可高效检测且不影响文本质量。
- **关键技术细节**:
  - 水印嵌入：
    1. 根据前一个token的哈希值，将词表随机划分为"绿色集"和"红色集"
    2. 在采样时，对绿色token增加偏置δ，提高其采样概率
  - 水印检测：
    1. 统计文本中绿色token的比例
    2. 使用z-test进行假设检验，计算p-value
    3. 仅需约25个token即可完成检测
  - 关键特性：检测算法可以公开，第三方（如社交媒体平台）可自行检测
- **实验设置**:
  - 模型：Open Pretrained Transformer (OPT) 多十亿参数模型
  - 评价指标：检测准确率、假阳性率（FPR=1e-5）、文本质量（困惑度）
  - 对比：与不加水印的文本对比
- **优点/局限**:
  - 优点：
    - 对文本质量影响极小
    - 检测高效，仅需少量token
    - 检测算法可公开，无需访问模型API
    - 信息论框架分析水印灵敏度
  - 局限：
    - 对改写攻击（paraphrasing）的鲁棒性有限
    - 需要访问模型的logits（修改采样过程）
- **我的思考/疑问**:
  - Green/Red list的设计非常优雅——通过token哈希确定颜色，保证随机性
  - 与音频水印的类比：green token的promotion类似于音频中的信号增强，red token的抑制类似于掩蔽
  - 这篇论文是Lai2024自适应文本水印的直接改进基础
- **关联文献**: [[Lai2024_Adaptive_Text_Watermark]], [[Reliability_LLM_Watermarks]]
- **代码/数据链接**: https://github.com/jwkirchenbauer/lm-watermarking
- **复现笔记**: （后续补充：环境、踩坑、调参、结果截图）
