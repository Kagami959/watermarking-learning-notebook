# DDPM 代码逐行讲解（第6-18部分）

## 第6部分：注意力模块（第197-261行）

### Attention 类（第201-227行）

```python
class Attention(nn.Module):
    """
    标准多头自注意力（用于图像特征图）
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5  # 缩放因子，防止点积过大导致梯度消失
        self.heads = heads
        hidden_dim = dim_head * heads  # 总隐藏维度 = 每个头的维度 × 头数
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)  # 用1×1卷积同时生成Q、K、V
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)  # 输出投影，将多头结果合并回原始维度

    def forward(self, x):
        b, c, h, w = x.shape  # batch, channels, height, width
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 将QKV在channel维度分成3份
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )  # 重排：将通道分成多头，空间维度展平
        q = q * self.scale  # Q乘以缩放因子

        sim = einsum("b h d i, b h d j -> b h i j", q, k)  # Q·K^T 计算相似度
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()  # 减去最大值（数值稳定性）
        attn = sim.softmax(dim=-1)  # softmax得到注意力权重

        out = einsum("b h i j, b h d j -> b h d i", attn, v)  # 加权求和V
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)  # 恢复空间维度
        return self.to_out(out)
```

**关键点：**
- 这是标准的自注意力，但针对图像做了适配：用1×1卷积代替线性层
- `einsum` 是爱因斯坦求和约定，高效计算矩阵乘法
- 空间维度 (h×w) 被展平处理，所以注意力在整个空间上计算

---

### LinearAttention 类（第230-261行）

```python
class LinearAttention(nn.Module):
    """
    线性注意力变体
    时间和内存复杂度与序列长度成线性关系（标准注意力是二次的）
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))  # 输出层加了GroupNorm

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)  # Q在channel维度softmax
        k = k.softmax(dim=-1)  # K在空间维度softmax

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)  # K·V^T 计算上下文

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)  # context·Q
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
```

**与标准注意力的关键区别：**
- **复杂度**：标准注意力 O(N²·d)，线性注意力 O(N·d²)
- **原理**：利用矩阵乘法结合律 `(Q·K^T)·V = Q·(K^T·V)`，先算 K·V 再与 Q 相乘
- Q和K都做了softmax，这是为了保证非负性，让分解更稳定

---

## 第7部分：Group Normalization 包装（第264-277行）

### PreNorm 类

```python
class PreNorm(nn.Module):
    """在注意力层前应用 Group Normalization"""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn  # 要包裹的模块（通常是注意力层）
        self.norm = nn.GroupNorm(1, dim)  # GroupNorm，num_groups=1 等价于 LayerNorm

    def forward(self, x):
        x = self.norm(x)  # 先归一化
        return self.fn(x)  # 再送入注意力层
```

**为什么用 GroupNorm(1, dim)：**
- 对图像任务，GroupNorm 比 BatchNorm 更稳定
- num_groups=1 时，等价于对整个 channel 做归一化
- "PreNorm" 是 Transformer 的常见技巧：先归一化再做注意力，比"后归一化"训练更稳定

---

## 第8部分：U-Net 模型（第280-425行）

### Unet 类

```python
class Unet(nn.Module):
    """
    条件 U-Net 模型
    输入: 带噪声的图像 (batch, channels, H, W) + 时间步 (batch,)
    输出: 预测的噪声 (batch, channels, H, W)
    """
    def __init__(
        self,
        dim,                    # 基础维度
        init_dim=None,          # 初始维度（默认等于dim）
        out_dim=None,           # 输出维度（默认等于channels）
        dim_mults=(1, 2, 4, 8), # 各层维度倍数
        channels=3,             # 输入图像通道数
        self_condition=False,   # 是否自条件化
        resnet_block_groups=4,  # ResNet块的GroupNorm组数
    ):
        super().__init__()

        # 确定维度
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)  # 自条件化时输入通道翻倍

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)  # 初始1×1卷积

        # 计算各层维度：[init_dim, dim*1, dim*2, dim*4, dim*8]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 配对相邻层：[(init_dim, dim), (dim, dim*2), (dim*2, dim*4), (dim*4, dim*8)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)  # 固定ResNet块的groups参数

        # 时间嵌入
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),  # 正弦位置编码
            nn.Linear(dim, time_dim),            # 线性变换
            nn.GELU(),                           # GELU激活
            nn.Linear(time_dim, time_dim),       # 再次线性变换
        )

        # 各层
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # 下采样阶段
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),  # ResNet块1
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),  # ResNet块2
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),  # 注意力（残差连接）
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                        # 最后一层用普通卷积，其他层用下采样
                    ]
                )
            )

        # 中间层
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))  # 中间层用标准注意力
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 上采样阶段
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),  # 输入维度是拼接后的
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # 最终输出
        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)  # 拼接初始残差
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
```

### 前向传播（第383-425行）

```python
    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)  # 自条件化：拼接上一次的预测

        x = self.init_conv(x)  # 初始卷积
        r = x.clone()  # 保存初始特征用于最后的跳跃连接

        t = self.time_mlp(time)  # 时间嵌入

        h = []  # 存储中间特征用于跳跃连接

        # 下采样
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)  # 第一个ResNet块（注入时间信息）
            h.append(x)  # 保存特征

            x = block2(x, t)  # 第二个ResNet块
            x = attn(x)  # 注意力
            h.append(x)  # 保存特征

            x = downsample(x)  # 下采样

        # 中间层
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 上采样（含跳跃连接）
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)  # 跳跃连接：拼接下采样时保存的特征
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)  # 另一个跳跃连接
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)  # 最终跳跃连接：拼接初始特征

        x = self.final_res_block(x, t)
        return self.final_conv(x)  # 输出预测的噪声
```

**U-Net 结构要点：**
1. **编码器-解码器结构**：下采样提取特征，上采样恢复分辨率
2. **跳跃连接**：每个下采样层的特征都保存，在上采样时拼接，保留空间细节
3. **时间条件化**：通过scale-shift机制将时间信息注入每个ResNet块
4. **不同层次用不同注意力**：下采样/上采样用线性注意力（效率高），中间层用标准注意力（表达能力强）

---

## 第9部分：噪声调度（第428-464行）

### 余弦调度

```python
def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度：比线性调度更平滑，在早期时间步添加噪声更慢
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    # 余弦函数：从1降到0，控制alpha_bar的衰减
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化，确保从1开始
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 从alpha_bar推导beta
    return torch.clip(betas, 0.0001, 0.9999)  # 裁剪防止极端值
```

### 其他调度方式

```python
def linear_beta_schedule(timesteps):
    """线性调度：beta从0.0001线性增长到0.02"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    """二次调度：beta按平方增长"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    """Sigmoid调度：S形曲线，两端平缓中间陡峭"""
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
```

**调度方式对比：**
- **线性**：最简单，但早期添加噪声太快
- **余弦**：更平滑，论文推荐，生成质量更好
- **二次/Sigmoid**：介于两者之间

---

## 第10部分：前向扩散过程相关变量（第467-494行）

```python
timesteps = 300  # 总扩散步数

# 定义beta调度
betas = linear_beta_schedule(timesteps=timesteps)

# alpha相关变量
alphas = 1. - betas  # alpha_t = 1 - beta_t
alphas_cumprod = torch.cumprod(alphas, axis=0)  # alpha_bar_t = ∏_{s=1}^t alpha_s（累积乘积）
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # alpha_bar_{t-1}，前面补1
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)  # 1/sqrt(alpha_t)

# 前向扩散 q(x_t | x_0) 相关计算
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # sqrt(alpha_bar_t)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)  # sqrt(1-alpha_bar_t)

# 后验 q(x_{t-1} | x_t, x_0) 相关计算
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
# 后验方差：用于采样时添加随机性


def extract(a, t, x_shape):
    """从一维张量a中，根据batch中每个样本的时间步t提取对应值"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())  # 按索引提取
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    # 重塑为 (batch_size, 1, 1, 1) 以便广播
```

**关键公式：**
- `alphas_cumprod` (α̅_t): 累积乘积，控制信号保留比例
- `sqrt_alphas_cumprod` * x_0: 原始信号部分
- `sqrt_one_minus_alphas_cumprod` * ε: 噪声部分
- `posterior_variance`: 后验分布的方差，控制采样时的随机性

---

## 第11部分：图像变换工具（第497-519行）

```python
image_size = 28      # MNIST图像尺寸
channels = 1         # 灰度图像
batch_size = 128

# 正向变换：PIL -> [-1, 1] 范围的tensor
transform = Compose([
    transforms.RandomHorizontalFlip(),           # 随机水平翻转（数据增强）
    transforms.ToTensor(),                        # 转为CHW tensor, 值域[0,1]
    transforms.Lambda(lambda t: (t * 2) - 1),     # 映射到[-1,1]（扩散模型常用）
])

# 逆向变换：[-1,1] tensor -> PIL图像
reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),                # 映射回[0,1]
    Lambda(lambda t: t.permute(1, 2, 0)),          # CHW -> HWC（matplotlib需要）
    Lambda(lambda t: t * 255.),                    # 映射到[0,255]
    Lambda(lambda t: t.numpy().astype(np.uint8)),  # 转为numpy数组
    ToPILImage(),                                 # 转为PIL图像
])
```

**为什么用[-1,1]范围：**
- 扩散模型通常假设数据在[-1,1]范围内
- 与噪声的分布（标准正态分布）匹配更好
- 训练更稳定

---

## 第12部分：前向扩散：q_sample（第522-546行）

```python
def q_sample(x_start, t, noise=None):
    """
    前向扩散过程（利用nice property直接采样）
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * ε
    这是DDPM的核心公式：直接从x_0计算任意时间步的x_t
    """
    if noise is None:
        noise = torch.randn_like(x_start)  # 生成标准正态噪声

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    # 线性组合：信号部分 + 噪声部分


def get_noisy_image(x_start, t):
    """获取指定时间步t的噪声图像（PIL格式）"""
    x_noisy = q_sample(x_start, t=t)
    noisy_image = reverse_transform(x_noisy.squeeze())  # squeeze去掉batch维度
    return noisy_image


def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    """可视化多张图像（用于展示不同时间步的噪声图像）"""
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(20, 5), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [None] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            if img is not None:
                ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=)

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
```

**q_sample 的数学意义：**
- 这是前向扩散的"nice property"：不需要逐步添加噪声，可以直接从x_0计算任意x_t
- 随着t增大，`sqrt_alphas_cumprod_t` 减小，`sqrt_one_minus_alphas_cumprod_t` 增大
- 最终 x_T ≈ 纯噪声（标准正态分布）

---

## 第13部分：损失函数（第572-598行）

```python
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    """
    计算去噪损失
    - 对x_start添加噪声得到x_noisy
    - 模型预测噪声
    - 计算预测噪声与真实噪声之间的损失
    """
    if noise is None:
        noise = torch.randn_like(x_start)  # 生成真实噪声

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)  # 添加噪声
    predicted_noise = denoise_model(x_noisy, t)  # 模型预测噪声

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)  # L1损失（平均绝对误差）
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)  # L2损失（均方误差）
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)  # Huber损失（L1和L2的平滑组合）
    else:
        raise NotImplementedError()

    return loss
```

**损失函数的选择：**
- **L1**：对异常值更鲁棒，但梯度不连续
- **L2**：梯度平滑，但对异常值敏感
- **Huber**：结合两者优点，小误差时用L2，大误差时用L1
- DDPM原论文用L1/L2都可以，这里用Huber是为了更稳定

---

## 第14部分：数据集加载（第600-621行）

```python
def get_dataloader():
    """加载Fashion-MNIST数据集并创建DataLoader"""
    from datasets import load_dataset

    # 从HuggingFace Hub加载数据集
    dataset = load_dataset("fashion_mnist")

    # 定义图像变换函数
    def transforms_fn(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        # 对每个图像应用变换：转灰度、转tensor、归一化到[-1,1]
        del examples["image"]  # 删除原始图像，只保留变换后的
        return examples

    transformed_dataset = dataset.with_transform(transforms_fn).remove_columns("label")
    # 移除标签列（无监督学习，不需要标签）
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    return dataloader
```

**Fashion-MNIST：**
- 28×28灰度图像，10个服装类别
- 替代MNIST的更现代数据集
- 这里只用图像，不用标签（扩散模型是无监督的）

---

## 第15部分：采样（推理）（第624-676行）

```python
@torch.no_grad()  # 推理时不需要计算梯度
def p_sample(model, x, t, t_index):
    """
    单步去噪采样（对应论文Algorithm 2的一步）
    使用模型预测的噪声来计算前一步的均值
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # 论文公式(11)：使用模型（噪声预测器）预测均值
    # μ_θ(x_t, t) = 1/sqrt(α_t) * (x_t - β_t/sqrt(1-α̅_t) * ε_θ(x_t, t))
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean  # 最后一步不需要加噪声
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 第4行：x_{t-1} = μ_θ + σ_t * z
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    """
    完整的采样循环（对应论文Algorithm 2）
    从纯噪声x_T开始，逐步去噪到x_0
    """
    device = next(model.parameters()).device

    b = shape[0]
    img = torch.randn(shape, device=device)  # 从纯噪声开始
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    """生成新图像的入口函数"""
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
```

**采样过程：**
1. 从纯高斯噪声开始（x_T）
2. 逐步去噪：x_T → x_{T-1} → ... → x_0
3. 每步用模型预测噪声，计算均值，添加少量随机性
4. 最后一步（t=0）不加随机性，直接返回均值

---

## 第16部分：训练（第679-741行）

```python
def train():
    """训练DDPM模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建结果目录
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000  # 每1000步保存一次生成的图像

    # 初始化模型
    model = Unet(
        dim=image_size,        # 28
        channels=channels,     # 1
        dim_mults=(1, 2, 4,)   # 维度倍数：28, 28, 56, 112
    )
    model.to(device)

    # 优化器
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 加载数据
    print("加载Fashion-MNIST数据集...")
    dataloader = get_dataloader()

    # 训练参数
    epochs = 6

    print(f"开始训练，共{epochs}个epoch...")
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size_curr = batch["pixel_values"].shape[0]
            batch_data = batch["pixel_values"].to(device)

            # Algorithm 1 第3行：对batch中每个样本均匀采样时间步t
            t = torch.randint(0, timesteps, (batch_size_curr,), device=device).long()

            loss = p_losses(model, batch_data, t, loss_type="huber")

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}")

            loss.backward()
            optimizer.step()

            # 定期保存生成的图像
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size_curr)
                all_images_list = list(map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels), batches))
                all_images = torch.cat([torch.from_numpy(al[-1]) for al in all_images_list], dim=0)
                all_images = (all_images + 1) * 0.5  # 从[-1,1]映射回[0,1]
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
                print(f"  已保存采样图像: sample-{milestone}.png")

    print("训练完成！")
    return model, device
```

**训练要点：**
1. **时间步采样**：每个batch随机采样时间步t，让模型学习所有时间步的去噪
2. **损失计算**：预测噪声 vs 真实噪声
3. **定期采样**：训练过程中定期生成图像，监控生成质量
4. **图像归一化**：训练用[-1,1]，保存时转回[0,1]

---

## 第17部分：推理 & 可视化（第744-777行）

```python
def inference(model, device):
    """训练后进行推理采样并可视化"""
    # 采样64张图像
    print("开始采样生成图像...")
    samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

    # 展示一张随机图像
    random_index = 5
    plt.figure(figsize=(4, 4))
    plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
    plt.title("Sampled Image")
    plt.axis("off")
    plt.savefig("sample_output.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("已保存采样图像: sample_output.png")

    # 创建去噪过程的GIF
    print("创建去噪过程GIF...")
    random_index = 53
    fig = plt.figure()
    ims = []
    for i in range(timesteps):
        im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels),
                        cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif', writer='pillow')
    plt.close()
    print("已保存去噪GIF: diffusion.gif")
```

**可视化内容：**
1. **单张生成图像**：展示最终生成结果
2. **去噪过程GIF**：展示从噪声到清晰图像的完整过程，直观理解扩散过程

---

## 第18部分：主入口（第780-786行）

```python
if __name__ == "__main__":
    model, device = train()      # 训练模型
    inference(model, device)     # 推理和可视化
```

**程序流程：**
1. 训练DDPM模型（6个epoch）
2. 使用训练好的模型生成新图像
3. 保存生成结果和去噪过程动画

---

## 总结

整个DDPM实现包含：
1. **模型架构**：U-Net（编码器-解码器 + 跳跃连接 + 时间条件化）
2. **注意力机制**：标准注意力（高质量）和线性注意力（高效）
3. **噪声调度**：多种调度方式，控制前向扩散过程
4. **训练过程**：随机时间步采样，预测噪声损失
5. **采样过程**：从纯噪声逐步去噪生成图像
