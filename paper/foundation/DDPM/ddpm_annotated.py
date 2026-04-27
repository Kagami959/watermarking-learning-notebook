"""
DDPM (Denoising Diffusion Probabilistic Models) 完整实现
基于 HuggingFace "The Annotated Diffusion Model" 博客复现
https://huggingface.co/blog/annotated-diffusion

参考论文: Ho et al., 2020 - "Denoising Diffusion Probabilistic Models"
参考实现: lucidrains/denoising-diffusion-pytorch
"""

# ============================================================
# 1. 导入依赖
# ============================================================
import math
from inspect import isfunction
from functools import partial
from pathlib import Path

import matplotlib 
matplotlib.use("Agg")  # 非交互式后端，适合无 GUI 环境；有桌面环境可改为 TkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import save_image


# ============================================================
# 2. 辅助函数
# ============================================================

def exists(x):
    """判断 x 是否不为 None"""
    return x is not None


def default(val, d):
    """如果 val 存在则返回 val，否则返回 d（d 可以是函数）"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    """将 num 拆分为若干个 divisor 组，余数单独一组"""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    """残差连接包装器: output = fn(x, ...) + x"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# ============================================================
# 3. 上/下采样模块
# ============================================================

def Upsample(dim, dim_out=None):
    """上采样: 最近邻放大 2x + 卷积"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """下采样: 空间重排 + 1x1 卷积（避免使用步幅卷积或池化）"""
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


# ============================================================
# 4. 位置编码（时间步嵌入）
# ============================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置嵌入，将时间步 t 编码为向量
    灵感来自 Transformer (Vaswani et al., 2017)
    输入: (batch_size,) 的时间步
    输出: (batch_size, dim) 的嵌入向量
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ============================================================
# 5. ResNet 块
# ============================================================

class WeightStandardizedConv2d(nn.Conv2d):
    """
    权重标准化卷积层
    参考: https://arxiv.org/abs/1903.10520
    权重标准化据说与 Group Normalization 协同效果更好
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    """基础卷积块: 权重标准化卷积 + GroupNorm + SiLU 激活，支持 scale_shift"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    ResNet 残差块
    参考: https://arxiv.org/abs/1512.03385
    包含时间嵌入的 scale-shift 调制
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ============================================================
# 6. 注意力模块
# ============================================================

class Attention(nn.Module):
    """
    标准多头自注意力（用于图像特征图）
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    线性注意力变体
    参考: https://arxiv.org/abs/1812.01243
    时间和内存复杂度与序列长度成线性关系（标准注意力是二次的）
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# ============================================================
# 7. Group Normalization 包装
# ============================================================

class PreNorm(nn.Module):
    """在注意力层前应用 Group Normalization"""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# ============================================================
# 8. U-Net 模型
# ============================================================

class Unet(nn.Module):
    """
    条件 U-Net 模型
    输入: 带噪声的图像 (batch, channels, H, W) + 时间步 (batch,)
    输出: 预测的噪声 (batch, channels, H, W)

    结构:
    - 初始卷积层
    - 下采样阶段: 多个 ResNet 块 + 注意力 + 下采样
    - 中间层: ResNet 块 + 注意力
    - 上采样阶段: 多个 ResNet 块 + 注意力 + 上采样（含跳跃连接）
    - 最终输出层
    """
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # 确定维度
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # 时间嵌入
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
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
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # 中间层
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 上采样阶段
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # 最终输出
        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        # 下采样
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # 中间层
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 上采样（含跳跃连接）
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# ============================================================
# 9. 噪声调度（Beta Schedule）
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度
    参考: https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """线性调度: beta 从 1e-4 线性增长到 0.02"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    """二次调度"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """Sigmoid 调度"""
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# ============================================================
# 10. 前向扩散过程相关变量
# ============================================================

timesteps = 300

# 定义 beta 调度
betas = linear_beta_schedule(timesteps=timesteps)

# 定义 alpha 相关变量
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# 前向扩散 q(x_t | x_{t-1}) 相关计算
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 后验 q(x_{t-1} | x_t, x_0) 相关计算
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    """从一维张量 a 中，根据 batch 中每个样本的时间步 t 提取对应值"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ============================================================
# 11. 图像变换工具
# ============================================================

image_size = 28
channels = 1
batch_size = 128

# 正向变换: PIL -> [-1, 1] 范围的 tensor
transform = Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 转为 CHW tensor, 值域 [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1),  # 映射到 [-1, 1]
])

# 逆向变换: [-1, 1] tensor -> PIL 图像
reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)),  # CHW -> HWC
    Lambda(lambda t: t * 255.),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),
])


# ============================================================
# 12. 前向扩散: q_sample
# ============================================================

def q_sample(x_start, t, noise=None):
    """
    前向扩散过程（利用 nice property 直接采样）
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
    """获取指定时间步 t 的噪声图像（PIL 格式）"""
    x_noisy = q_sample(x_start, t=t)
    noisy_image = reverse_transform(x_noisy.squeeze())
    return noisy_image


def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    """可视化多张图像"""
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
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


# ============================================================
# 13. 损失函数
# ============================================================

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    """
    计算去噪损失
    - 对 x_start 添加噪声得到 x_noisy
    - 模型预测噪声
    - 计算预测噪声与真实噪声之间的损失
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# ============================================================
# 14. 数据集加载
# ============================================================

def get_dataloader():
    """加载 Fashion-MNIST 数据集并创建 DataLoader"""
    from datasets import load_dataset

    # 从 HuggingFace Hub 加载数据集
    dataset = load_dataset("fashion_mnist")

    # 定义图像变换函数
    def transforms_fn(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]
        return examples

    transformed_dataset = dataset.with_transform(transforms_fn).remove_columns("label")
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    return dataloader


# ============================================================
# 15. 采样（推理）
# ============================================================

@torch.no_grad()
def p_sample(model, x, t, t_index):
    """
    单步去噪采样（对应论文 Algorithm 2 的一步）
    使用模型预测的噪声来计算前一步的均值
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # 论文公式 (11): 使用模型（噪声预测器）预测均值
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 第 4 行
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    """
    完整的采样循环（对应论文 Algorithm 2）
    从纯噪声 x_T 开始，逐步去噪到 x_0
    """
    device = next(model.parameters()).device

    b = shape[0]
    # 从纯噪声开始
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    """生成新图像的入口函数"""
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


# ============================================================
# 16. 训练
# ============================================================

def train():
    """训练 DDPM 模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建结果目录
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000

    # 初始化模型
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    # 优化器
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 加载数据
    print("加载 Fashion-MNIST 数据集...")
    dataloader = get_dataloader()

    # 训练参数
    epochs = 6

    print(f"开始训练，共 {epochs} 个 epoch...")
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size_curr = batch["pixel_values"].shape[0]
            batch_data = batch["pixel_values"].to(device)

            # Algorithm 1 第 3 行: 对 batch 中每个样本均匀采样时间步 t
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
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
                print(f"  已保存采样图像: sample-{milestone}.png")

    print("训练完成！")
    return model, device


# ============================================================
# 17. 推理 & 可视化
# ============================================================

def inference(model, device):
    """训练后进行推理采样并可视化"""
    # 采样 64 张图像
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

    # 创建去噪过程的 GIF
    print("创建去噪过程 GIF...")
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
    print("已保存去噪 GIF: diffusion.gif")


# ============================================================
# 18. 主入口
# ============================================================

if __name__ == "__main__":
    model, device = train()
    inference(model, device)
