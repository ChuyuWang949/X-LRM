# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Modified by Zexin He in 2023-2024.
# The modifications are subject to the same license as the original.
# Further modified in X-LRM for CBCT reconstruction.

import torch

class RaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None

    @torch.compile
    def forward(self, cam2world_matrix, intrinsics, resolutions, anchors, region_size):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolutions: (N, 1)
        anchors: (N, 2)
        region_size: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """

        N, M = cam2world_matrix.shape[0], region_size**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = torch.stack(torch.meshgrid(
            torch.arange(region_size, dtype=torch.float32, device=cam2world_matrix.device),
            torch.arange(region_size, dtype=torch.float32, device=cam2world_matrix.device),
            indexing='ij',
        ))
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        # anchors are indexed as normal (row, col) but uv is indexed as (x, y)
        x_cam = (uv[:, :, 0].view(N, -1) + anchors[:, 1].unsqueeze(-1)) * (1./resolutions) + (0.5/resolutions)
        y_cam = (uv[:, :, 1].view(N, -1) + anchors[:, 0].unsqueeze(-1)) * (1./resolutions) + (0.5/resolutions)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        # Lift OpenCV pixel coordinates into the Blender camera basis.
        # Rotate directions directly so AMP cannot round a translated point
        # before subtraction of the camera origin.
        camera_dirs = torch.stack((x_lift, -y_lift, -z_cam), dim=-1)
        with torch.autocast(device_type=cam2world_matrix.device.type, enabled=False):
            rotation = cam2world_matrix[:, :3, :3].to(camera_dirs.dtype)
            ray_dirs = torch.nn.functional.normalize(
                torch.bmm(camera_dirs, rotation.transpose(1, 2)), dim=-1
            )
        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs
