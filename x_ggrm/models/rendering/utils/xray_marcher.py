import torch.nn as nn
import torch

class CTRayMarcher(nn.Module):
    """
    一个用于CT风格渲染的Ray Marcher模块。

    这个模块不处理RGB颜色，而是直接对网络预测的密度值沿着光线进行积分，
    得到一个单一的累积密度值（类似于CT图像中的亨氏单位 Hounsfield Unit）。
    这个过程模拟了X射线穿过物体时的衰减累积。

    它还会返回一个权重张量，用于指导精细化采样。
    """

    def __init__(self, activation_factory):
        """
        初始化模块。

        Args:
            activation_factory (function): 一个工厂函数，它接收 rendering_options 字典
            并返回一个PyTorch激活函数 (例如 F.relu)。
        """
        super().__init__()
        self.activation_factory = activation_factory

    def forward(self, densities, depths, rendering_options=None):
        """
        执行前向传播，使用NeRF风格的体积渲染计算X光投影。

        Args:
            densities (torch.Tensor):
            神经网络为每个采样点预测的原始密度值。
            形状: (B, R, S, 1)，其中 B=批大小, R=光线数, S=每条光线的采样点数。
            depths (torch.Tensor):
            每条光线上每个采样点的深度值。
            形状: (B, R, S, 1)。
            rendering_options (dict, optional): 包含渲染选项的字典。 Defaults to None.

        Returns:
            tuple:
                - composite_accumulation (torch.Tensor): 每条光线的最终累积密度值。形状: (B, R, 1)。
                - weights (torch.Tensor): 用于精细化采样的权重。形状: (B, R, S-1)。
        """
        if rendering_options is None:
            rendering_options = {}

        # --- 1. NeRF式体积渲染用于X光投影 ---

        # 1a. 计算采样间距 (类似NeRF的dists计算)
        # 形状从 (B, R, S, 1) -> (B, R, S-1, 1)
        dists = depths[:, :, 1:] - depths[:, :, :-1]
        
        # 1b. 处理最后一个采样点的距离 (NeRF标准做法)
        # 添加一个小的距离值避免边界问题
        last_dist = torch.full_like(depths[:, :, -1:], 1e-10)
        dists = torch.cat([dists, last_dist], dim=-2)  # 形状: (B, R, S, 1)

        activation_func = self.activation_factory(rendering_options)
        densities_activated = activation_func(densities)
        
        # 1d. 添加噪声用于正则化 (可选，类似NeRF的raw_noise_std)
        noise_std = rendering_options.get('raw_noise_std', 0.0)
        if noise_std > 0.0 and self.training:
            noise = torch.randn_like(densities_activated) * noise_std
            densities_activated = densities_activated + noise
        
        # 1e. X光投影积分 (对应NeRF的acc计算)
        # P(pixel) = ∫ μ(t) dt = Σ μ_i * Δt_i
        density_contributions = densities_activated * dists  # 形状: (B, R, S, 1)
        composite_accumulation = torch.sum(density_contributions, dim=-2)  # 形状: (B, R, 1)

        # 1f. 防止数值问题并确保物理一致性
        composite_accumulation = torch.nan_to_num(composite_accumulation, 1.0)
        composite_accumulation = torch.clamp(composite_accumulation, min=0.0)  # X光投影不能为负

        # --- 2. 计算重要性采样权重 (类似NeRF的weights) ---
        # 基于密度变化和数值大小计算权重，用于hierarchical sampling
        
        # 方法1: 基于密度梯度 (密度变化大的地方需要更细采样)
        density_grad_weights = torch.abs(densities_activated[:, :, 1:] - densities_activated[:, :, :-1])
        
        # 方法2: 基于密度大小 (高密度区域需要更细采样)
        density_mag_weights = (densities_activated[:, :, :-1] + densities_activated[:, :, 1:]) / 2
        
        # 组合权重策略
        alpha = rendering_options.get('weight_alpha', 0.7)  # 梯度权重系数
        beta = rendering_options.get('weight_beta', 0.3)    # 幅度权重系数
        
        weights = alpha * density_grad_weights.squeeze(-1) + beta * density_mag_weights.squeeze(-1)
        
        # 归一化权重并添加小的epsilon避免零权重
        weights = weights + 1e-8
        weights_sum = torch.sum(weights, dim=-1, keepdim=True) + 1e-8
        weights = weights / weights_sum

        return composite_accumulation, weights
