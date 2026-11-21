# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for X-LRM for CBCT reconstruction.

import torch
import torch.nn as nn
from accelerate.logging import get_logger

from x_ggrm.models.lrm.transformer import TransformerDecoder
from .rendering.synthesizer import TriplaneSynthesizer


logger = get_logger(__name__)


class ModelLRM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self,
        rendering_samples_per_ray: int,
        image_res: int,
        encoder_layers: int,
        encoder_heads: int,
        encoder_dim: int,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        triplane_low_res: int,
        triplane_high_res: int,
        triplane_dim: int,
        encoder_freeze: bool = True,
        encoder_type: str = 'dino',
        encoder_model_name: str = 'facebook/dino-vitb16',
        encoder_feat_dim: int = 768,
        volume_grid_size: int = 256,
        view_encoder_dropout: float | None = None,
        patch_pos_embed: bool = False,
        point_decoder_hidden_dim: int = 128,
        ) -> None:
        super().__init__()
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.volume_grid_size = volume_grid_size
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim
        self._encoder_type = encoder_type  # Store encoder type for forward logic

        # modules
        if encoder_type.startswith('patchify_linear') or encoder_type.startswith('adaptive_patchify_linear'):
            # P&L encoder for dynamic multi-view input [V_i, H, W, 7]
            self.encoder = self._encoder_fn(encoder_type)(
                input_channels=7,              # 1 image + 6 RPPC channels
                image_size=image_res,                # Input image size
                patch_size=16,                 # ViT-B/16 patch size
                embed_dim=encoder_feat_dim,    # d_E = encoder_feat_dim (384)
                use_pos_embed=patch_pos_embed,
                dropout=0.1,
                view_encoder_dropout=view_encoder_dropout,
            )
        else:
            # Standard single-image encoder
            self.encoder = self._encoder_fn(encoder_type)(
                model_name=encoder_model_name,
                freeze=encoder_freeze,
            )

        self.image_encoder = TransformerDecoder(
            block_type='basic',
            num_layers=encoder_layers, num_heads=encoder_heads,
            inner_dim=encoder_dim,
        )

        # CRITICAL FIX: Use proper positional embedding initialization
        # Standard practice: use smaller scale but not too small to avoid vanishing gradients
        pos_embed_scale = 0.02  # Similar to ViT/BERT practices, much larger than 0.031
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * pos_embed_scale)
        self.transformer = TransformerDecoder(
            block_type='cond',
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim, cond_dim=encoder_feat_dim,
        )
        if triplane_low_res <= 0 or triplane_high_res < triplane_low_res or triplane_high_res % triplane_low_res:
            raise ValueError('High triplane resolution must be a positive integer multiple of low resolution')
        upsample_factor = triplane_high_res // triplane_low_res
        self.upsampler = nn.ConvTranspose2d(
            transformer_dim, triplane_dim, kernel_size=upsample_factor, stride=upsample_factor
        )
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim, samples_per_ray=rendering_samples_per_ray,
            point_decoder_hidden_dim=point_decoder_hidden_dim,
        )

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        supported_types = ['dino', 'dinov2', 'patchify_linear', 'adaptive_patchify_linear']
        assert encoder_type in supported_types, f"Unsupported encoder type. Choose from: {supported_types}"
        
        if encoder_type == 'dino':
            from .encoders.dino_wrapper import DinoWrapper
            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == 'dinov2':
            from .encoders.dinov2_wrapper import Dinov2Wrapper
            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper
        elif encoder_type == 'patchify_linear':
            from .encoders.patchify_linear_encoder import create_patchify_linear_encoder
            logger.info("Using Patchify & Linear (P&L) encoder")
            return lambda **kwargs: create_patchify_linear_encoder('patchify_linear', **kwargs)
        return None

    def forward_transformer(self, image_feats):
        N = image_feats.shape[0]
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        x = self.transformer(
            x,
            cond=image_feats
        )
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    def forward_planes(self, image):
        """
        Forward pass to generate triplane features
        
        Args:
            image: Input images in various formats:
                   - [N, C_img, H_img, W_img]: Single image per batch (legacy)
                   - [N, M, C_img, H_img, W_img]: Multiple images per batch (multi-image)
                   - [N, V_i, H, W, 7]: Dynamic multi-view with RPPC (P&L encoder)
            
        Returns:
            planes: [N, 3, H', W', D'] Triplane features
        """
        # Handle different input formats based on encoder type
        encoder_type = getattr(self, '_encoder_type', 'unknown')
        
        if encoder_type == 'patchify_linear':
            # P&L encoder: expects [N, V_i, H, W, 7]
            if len(image.shape) == 5 and image.shape[-1] == 7:
                N, V_i, H, W, C = image.shape
                # Reshape to process all views in parallel: [N*V_i, H, W, 7]
                image_reshaped = image.view(N * V_i, H, W, C)
                
                # Process all views in parallel
                all_feats = self.encoder(image_reshaped)  # [N*V_i*tokens_per_view, embed_dim]
                
                # Reshape back to batch dimension: [N, V_i*tokens_per_view, embed_dim]
                tokens_per_view = all_feats.shape[0] // (N * V_i)
                image_feats = all_feats.view(N, V_i * tokens_per_view, -1)
                image_feats = self.image_encoder(image_feats)
            else:
                raise ValueError(f"P&L encoder expects [N, V_i, H, W, 7] input, got {image.shape}")
        else:
            # Legacy encoders: [N, C_img, H_img, W_img] or [N, M, C_img, H_img, W_img]
            image_feats = self.encoder(image)  # Multi-image encoder handles both single and multi-image inputs
        
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # 优化：使用checkpointing减少内存使用，不改变模型结构
        N = image_feats.shape[0]

        # transformer generating planes - 训练时使用checkpointing节省内存
        if self.training:
            tokens = torch.utils.checkpoint.checkpoint(
                self.forward_transformer, image_feats, use_reentrant=False
            )
        else:
            tokens = self.forward_transformer(image_feats)
        planes = self.reshape_upsample(tokens)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes

    def forward(self, image, render_cameras=None, render_anchors=None, render_resolutions=None,
                render_region_size: int | None = None, *, ct_query_points=None):
        """
        Forward pass through the LRM model
        
        Args:
            image: Input images in various formats:
                   - [N, C_img, H_img, W_img]: Single image per batch (legacy)
                   - [N, M, C_img, H_img, W_img]: Multiple images per batch (multi-image)
                   - [N, V_i, H, W, 7]: Dynamic multi-view with RPPC (P&L encoder)
            render_cameras: [N, M, D_cam_render] Render camera parameters
            render_anchors: [N, M, 2] Render anchors
            render_resolutions: [N, M, 1] Render resolutions
            render_region_size: int Render region size
            
        Returns:
            dict: {
                'planes': [N, 3, H', W', D'] triplane features,
                'vol_pred': [N, D, H, W] predicted volume,
                'images_density': [N, M, 1, H, W] rendered density images
            }
        """
        planes = self.forward_planes(image)
        if ct_query_points is not None:
            densities = self.synthesizer.forward_points(planes, ct_query_points)['sigma']
            return {'planes': planes, 'ct_point_pred': densities}
        if render_cameras is None:
            density = self.synthesizer.forward_grid(planes, grid_size=self.volume_grid_size)['sigma']
            return {'planes': planes, 'vol_pred': density.squeeze(-1)}

        assert image.shape[0] == render_cameras.shape[0], "Batch size mismatch for image and render_cameras"
        assert image.shape[0] == render_anchors.shape[0], "Batch size mismatch for image and render_anchors"
        N, M = render_cameras.shape[:2]

        # render target views
        render_results = self.synthesizer(planes, render_cameras, render_anchors, render_resolutions, render_region_size)
        assert render_results['images_density'].shape[0] == N, "Batch size mismatch for render_results"
        assert render_results['images_density'].shape[1] == M, "Number of rendered views should be consistent with render_cameras"

        vol_features = self.synthesizer.forward_grid(planes, grid_size=self.volume_grid_size)

        # CRITICAL FIX: Remove double sigmoid activation!
        # vol_features['sigma'] already has sigmoid applied in run_model_activated
        vol_pred = vol_features['sigma'].squeeze(-1)  # Already in [0,1] range

        return {
            'planes': planes,
            'vol_pred': vol_pred,
            **render_results,
        }
