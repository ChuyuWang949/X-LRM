"""Geometry-aligned local image features for XLRM point queries.

Cameras use XLRM's existing 25-value format: Blender world_from_camera
(row-major 4x4), followed by image-normalized OpenCV intrinsics (3x3).
Points and camera translations must use the same world units.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ReprojectionEncoder(nn.Module):
    """Learn four image scales from projection intensity, independently per view."""

    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        self.channels = tuple(channels)
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError('Reprojection channels must be positive')
        self.stages = nn.ModuleList()
        previous = 1
        for level, channel in enumerate(self.channels):
            layers = []
            if level:
                layers.extend([nn.Conv2d(previous, previous, 2, stride=2),
                               nn.BatchNorm2d(previous), nn.LeakyReLU()])
            layers.extend([nn.Conv2d(previous, channel, 3, padding=1),
                           nn.BatchNorm2d(channel), nn.LeakyReLU(),
                           nn.Conv2d(channel, channel, 3, padding=1),
                           nn.BatchNorm2d(channel), nn.LeakyReLU()])
            self.stages.append(nn.Sequential(*layers))
            previous = channel

    def forward(self, projections):
        # projections: [B, V, 1, H, W], with the same intensity as the LRM input.
        batch, views = projections.shape[:2]
        x = projections.flatten(0, 1)
        maps = []
        for stage in self.stages:
            x = stage(x)
            maps.append(x.reshape(batch, views, *x.shape[1:]))
        return maps


def project_points(points, cameras):
    """Return conventional (column, row) grid_sample coordinates and visibility.

    Intrinsics map rays to image fractions, so grid=2*uv-1 is pixel-centred
    with align_corners=False. Geometry stays FP32 even under BF16 autocast.
    """
    with torch.autocast(device_type=points.device.type, enabled=False):
        poses = cameras[..., :16].float().reshape(*cameras.shape[:2], 4, 4)
        intrinsic = cameras[..., 16:25].float().reshape(*cameras.shape[:2], 3, 3)
        delta = points.float()[:, None] - poses[..., :3, 3].unsqueeze(2)
        # Explicit 3-term sums also avoid TF32 truncation of FP32 matmuls.
        local = (delta.unsqueeze(-1) * poses[..., None, :3, :3]).sum(dim=-2)
        local = local * local.new_tensor([1., -1., -1.])  # Blender -> OpenCV
        pixels_h = (intrinsic[:, :, None] * local.unsqueeze(-2)).sum(dim=-1)
        depth = local[..., 2]
        uv = pixels_h[..., :2] / pixels_h[..., 2:].clamp_min(1e-8)
        grid = uv * 2 - 1
        valid = (depth > 1e-8) & (grid.abs() <= 1).all(dim=-1)
    return grid, valid


def sample_reprojection_features(feature_maps, points, cameras, view_mask=None):
    """Bilinearly sample each scale, max-pool valid views, concatenate channels."""
    grid, valid = project_points(points, cameras)
    if view_mask is not None:
        valid = valid & view_mask.bool().unsqueeze(-1)
    batch, views, count = grid.shape[:3]
    # Invalid geometry is never sent to grid_sample with extreme coordinates.
    sampling_grid = grid.masked_fill(~valid[..., None], 2).reshape(batch * views, count, 1, 2)
    result = []
    for maps in feature_maps:
        sampled = F.grid_sample(maps.flatten(0, 1), sampling_grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        sampled = sampled.reshape(batch, views, maps.shape[2], count)
        sampled = sampled.masked_fill(~valid[:, :, None], -torch.inf)
        pooled = sampled.max(dim=1).values
        pooled = torch.where(valid.any(dim=1)[:, None], pooled, torch.zeros_like(pooled))
        result.append(pooled.transpose(1, 2))
    return torch.cat(result, dim=-1)  # [B, P, sum(channels)]
