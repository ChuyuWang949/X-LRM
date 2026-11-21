import torch
import torch.nn as nn

__all__ = ['TV3DLoss']

class TV3DLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算3D TV Loss。
        假设输入张量的形状为 [B, C, D, H, W]，其中 B 是批次，C 是通道，
        D, H, W 分别是深度、高度和宽度。

        Args:
            x (torch.Tensor): 输入的五维张量，形状为 [B, C, D, H, W]。

        Returns:
            torch.Tensor: 一个标量张量，代表计算出的TV loss的平均值。
        """
        # 确保输入至少是三维的（D, H, W）
        if x.dim() < 3:
            raise ValueError("Input tensor for TV3DLoss must have at least 3 dimensions (D, H, W)")

        # 为了方便处理，我们将批次和通道维度合并
        # 原始形状: [B, C, D, H, W] -> [B*C, D, H, W]
        # 这样我们可以统一处理每个3D体
        original_shape = x.shape
        x = x.reshape(-1, *original_shape[-3:])  # [N, D, H, W] where N = B*C

        # 计算三个空间维度上的梯度（相邻元素之差）
        # 沿深度(D)方向的差异
        diff_d = x[:, 1:, :, :] - x[:, :-1, :, :]
        # 沿高度(H)方向的差异
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        # 沿宽度(W)方向的差异
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]

        # 计算每个方向上差异的平方和的平均值
        tv_d = torch.mean(diff_d.pow(2))
        tv_h = torch.mean(diff_h.pow(2))
        tv_w = torch.mean(diff_w.pow(2))

        # 将三个方向的TV loss相加，得到最终的总TV loss
        total_tv_loss = tv_d + tv_h + tv_w

        return total_tv_loss