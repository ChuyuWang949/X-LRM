import torch
import torch.nn as nn
from accelerate.logging import get_logger


logger = get_logger(__name__)


class PatchifyLinearEncoder(nn.Module):
    """
    Patchify and Linear (P&L) Encoder for Dynamic Multi-View Processing
    
    Process flow:
    1. Input: [Vi, H, W, 7] where V_i is variable number of views
    2. Patchify: Split each 256x256 input into 16x16 patches -> [Vi, 256, 16*16*7]
    3. Linear: Project each patch to d_E=384 dimensions -> [Vi, 256, 384]
    4. Fusion: Concatenate all view tokens -> [Vi*256, 384] (no CLS token)
    """
    
    def __init__(
        self,
        input_channels: int = 7,           # 1 image + 6 RPPC channels
        image_size: int = 256,             # Input image size
        patch_size: int = 16,              # Patch size (ViT-B/16)
        embed_dim: int = 384,              # d_E embedding dimension
        use_pos_embed: bool = True,        # Whether to use positional embedding
        dropout: float = 0.1,
        view_encoder_dropout: float | None = None,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate patch dimensions
        assert image_size % patch_size == 0, f"Image size {image_size} must be divisible by patch size {patch_size}"
        self.num_patches_per_side = image_size // patch_size  # 256 // 16 = 16
        self.num_patches = self.num_patches_per_side ** 2     # 16 * 16 = 256
        self.patch_dim = patch_size * patch_size * input_channels  # 16 * 16 * 7 = 1792
        
        if use_pos_embed:
            # One detector-row/column embedding shared by all views.
            side = self.num_patches_per_side
            self.patch_pos_embed = nn.Parameter(torch.zeros(1, side, side, embed_dim))
        else:
            self.register_parameter('patch_pos_embed', None)

        # Patchify and Linear projection
        self.patch_embedding = nn.Sequential(
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # No CLS token - simplified approach
        self.tokens_per_view = self.num_patches      # 256
        
        # View-wise processing (optional)
        self.view_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=dropout if view_encoder_dropout is None else view_encoder_dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"PatchifyLinearEncoder initialized (No CLS Token):")
        logger.info(f"  - Input: [V_i, {image_size}, {image_size}, {input_channels}]")
        logger.info(f"  - Patches: {self.num_patches} patches per view")
        logger.info(f"  - Patch dim: {self.patch_dim} -> Embed dim: {embed_dim}")
        logger.info(f"  - Tokens per view: {self.tokens_per_view}")
        logger.info(f"  - Output: [V_i * {self.tokens_per_view}, {embed_dim}]")
    
    def _init_weights(self):
        """Initialize weights following ViT initialization"""
        # Patch embedding initialization
        nn.init.xavier_uniform_(self.patch_embedding[0].weight.view([self.patch_embedding[0].weight.shape[0], -1]))
        nn.init.normal_(self.patch_embedding[0].bias, std=1e-6)
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patchify input tensor
        
        Args:
            x: [V_i, H, W, C] input tensor
            
        Returns:
            patches: [V_i, num_patches, patch_dim] patched tensor
        """
        V_i, H, W, C = x.shape
        assert H == W == self.image_size, f"Expected image size {self.image_size}, got {H}x{W}"
        assert C == self.input_channels, f"Expected {self.input_channels} channels, got {C}"
        
        # Reshape to [V_i, C, H, W] for unfold operation
        x = x.permute(0, 3, 1, 2)  # [V_i, C, H, W]
        
        # Extract patches using unfold
        # unfold(dimension, size, step) -> [V_i, C, num_patches_H, patch_size, num_patches_W, patch_size]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Reshape to [V_i, num_patches, patch_dim]
        patches = patches.contiguous().view(
            V_i, 
            C,
            self.num_patches_per_side,
            self.num_patches_per_side, 
            self.patch_size,
            self.patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [V_i, num_patches_H, num_patches_W, C, patch_size, patch_size]
        patches = patches.contiguous().view(V_i, self.num_patches, self.patch_dim)  # [V_i, num_patches, patch_dim]
        
        return patches
    
    def forward(self, input_views: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through P&L encoder
        
        Args:
            input_views: [V_i, H, W, 7] dynamic multi-view input with RPPC
            
        Returns:
            fused_tokens: [V_i * num_patches, embed_dim] fused patch tokens
        """
        V_i, H, W, C = input_views.shape
        
        # Step 1: Patchify - Split into patches
        patches = self.patchify(input_views)  # [V_i, num_patches, patch_diview
        
        # Step 2: Linear - Project patches to embedding space
        view_tokens = self.patch_embedding(patches)  # [V_i, num_patches, embed_dim]
        if self.patch_pos_embed is not None:
            positions = self.patch_pos_embed.flatten(1, 2).to(view_tokens.dtype)
            view_tokens = view_tokens + positions

        # Step 4: View-wise processing (per-view transformer)
        processed_tokens = []
        for i in range(V_i):
            view_token = view_tokens[i:i+1]  # [1, num_patches, embed_dim]
            processed_token = self.view_encoder(view_token)  # [1, num_patches, embed_dim]
            processed_tokens.append(processed_token.squeeze(0))  # [num_patches, embed_dim]
        
        # Step 5: Fusion - Concatenate all view tokens
        fused_tokens = torch.cat(processed_tokens, dim=0)  # [V_i * num_patches, embed_dim]
        
        logger.debug(f"P&L Encoder: {input_views.shape} -> {fused_tokens.shape}")
        logger.debug(f"  Views: {V_i}, Patches per view: {self.num_patches}, Total tokens: {fused_tokens.shape[0]}")
        
        return fused_tokens

# Factory function for easy integration
def create_patchify_linear_encoder(encoder_type: str = 'patchify_linear', **kwargs):
    """
    Factory function to create P&L encoders

    Args:
        encoder_type: 'patchify_linear' or 'adaptive_patchify_linear'
        **kwargs: encoder parameters

    Returns:
        PatchifyLinearEncoder instance
    """
    if encoder_type == 'patchify_linear':
        return PatchifyLinearEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")