import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, List

from .reader import sceneLoadTypeCallbacks
from .base import BaseDataset
from x_ggrm.utils.camera_utils import build_camera_standard
from .optimized_reader import register_optimized_reader

# 确保优化数据读取器已注册
register_optimized_reader()


@dataclass
class CTLRMDatasetCfg:
    root_dirs: List[str]
    meta_path: str
    split: Literal['train', 'val'] = 'train'

    # --- 数据分割参数 ---
    train_nums: int = 20000  # train数据占比，剩余的作为val数据
    random_seed: int = 42  # 用于保证分割的一致性

    # --- 动态多视图采样参数 ---
    input_image_res: int = 256
    nums_input_view: int = 10
    nums_target_views: int = 1
    # --- 图像处理与增强参数 ---
    render_image_res_low: int = 64
    render_image_res_high: int = 192
    render_region_size: int = 64

    device: Literal['cpu', 'cuda'] = 'cpu'


class CTLRMDataset(BaseDataset):
    def __init__(self, cfg: CTLRMDatasetCfg):
        super().__init__(cfg.root_dirs, cfg.meta_path)
        self.cfg = cfg
        
        # 根据split分割scenes，确保train和val使用完全不同的scene
        self.ids = self._split_scenes_by_split(self.ids, cfg.split, cfg.train_nums, cfg.random_seed)
        
        print(f"[{self.__class__.__name__}] Initialized for split '{cfg.split}' with {len(self.ids)} scenes.")

    def _split_scenes_by_split(self, all_ids: List[str], split: str, train_nums: int, seed: int) -> List[str]:
        # 设置随机种子确保分割的一致性
        rng = np.random.RandomState(seed)
        
        # 对所有scenes进行随机打乱
        shuffled_ids = all_ids.copy()
        rng.shuffle(shuffled_ids)
        
        # 计算train数据的数量
        num_total = len(shuffled_ids)
        num_train = train_nums
        
        # 分割数据
        if split == 'train':
            split_ids = shuffled_ids[:num_train]
        elif split == 'val':
            split_ids = shuffled_ids[num_train:]
        else:
            raise ValueError(f"Unknown split: {split}")
            
        print(f"[{self.__class__.__name__}] Split '{split}': {len(split_ids)}/{num_total} scenes")
        return split_ids

    def __len__(self) -> int:
        return len(self.ids)

    def inner_get_item(self, idx: int) -> dict:
        scene_id = self.ids[idx]

        # 检查是否为优化格式数据
        root_dir = self._locate_datadir(self.cfg.root_dirs, scene_id, "metadata.json")
        # 使用优化格式
        scene_dir = os.path.join(root_dir, scene_id)
        scene_info = sceneLoadTypeCallbacks["OptimizedBlender"](scene_dir)
        return self._handle_optimized_format(scene_info, scene_id)

    def _handle_optimized_format(self, scene_info, scene_id: str) -> dict:
        """处理优化格式数据"""
        # 动态选择视图数量配置
        num_input_views = self.cfg.nums_input_view
        num_target_views = self.cfg.nums_target_views

        # 获取总可用视图数量
        total_views = scene_info.all_camera_tensors['images'].shape[0]

        # 选择输入视图索引
        selected_indices_input = np.random.choice(total_views,
                                          min(num_input_views, total_views),
                                          replace=False)
        
        # 选择目标视图索引：优先从输入视图中选取
        if num_target_views <= len(selected_indices_input):
            # 目标视图数量不超过输入视图，直接从输入视图中选取
            selected_indices_target = np.random.choice(selected_indices_input,
                                              num_target_views,
                                              replace=False)
        else:
            # 目标视图数量超过输入视图，先选择所有输入视图，再从剩余中补充
            remaining_indices = np.setdiff1d(np.arange(total_views), selected_indices_input)
            additional_needed = num_target_views - len(selected_indices_input)
            
            if len(remaining_indices) >= additional_needed:
                additional_indices = np.random.choice(remaining_indices,
                                                    additional_needed,
                                                    replace=False)
                selected_indices_target = np.concatenate([selected_indices_input, additional_indices])
            else:
                # 剩余视图不够，使用所有可用视图
                selected_indices_target = np.arange(total_views)

        # 处理输入视图：从预计算的数据中提取
        input_data = self._package_input_views(scene_info.all_camera_tensors, selected_indices_input)
        target_data = self._package_target_data(scene_info.all_camera_tensors, selected_indices_target)

        return {
            'id': scene_id,
            'input_views': input_data,           # [V_i, H, W, 7]
            'render_cameras': target_data['cameras'],            
            'render_images': target_data['images'],              
            'render_anchors': target_data['anchors'],            
            'render_full_resolutions': target_data['resolutions'], 
            'vol_gt': scene_info.vol,
        }

    def _package_input_views(self, camera_tensors: dict, selected_indices: np.ndarray) -> dict:
        """
        处理优化格式的输入视图，生成 X ∈ R^{V_i × H × W × 7} 格式
        """
        H, W = self.cfg.input_image_res, self.cfg.input_image_res

        # 从预计算的数据中提取选定的视图
        selected_images = camera_tensors['images'][selected_indices]  # [V_i, 1, H_orig, W_orig]
        selected_rppc = camera_tensors['rppc_params'][selected_indices]  # [V_i, H_orig, W_orig, 6]

        # 只resize图像，RPPC参数保持原始分辨率下的几何意义
        if selected_images.shape[-1] != W or selected_images.shape[-2] != H:
            selected_images = F.interpolate(
                selected_images,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        selected_rppc = selected_rppc.permute(0, 3, 1, 2)
        if selected_rppc.shape[-1] != W or selected_rppc.shape[-2] != H:
            selected_rppc = F.interpolate(
                selected_rppc,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        selected_rppc = selected_rppc.permute(0, 2, 3, 1)

        # 拼接图像和RPPC参数
        input_tensor = torch.cat([selected_images.squeeze(1).unsqueeze(-1), selected_rppc], dim=-1)  # [V_i, H, W, 7]
        return input_tensor

    def _package_target_data(self, camera_tensors: dict, selected_indices: np.ndarray) -> dict:
        """处理优化格式的目标数据，使用standard标准相机格式"""
        from x_ggrm.utils.camera_utils import build_camera_standard
        
        num_views = len(selected_indices)
        
        # 同一个场景的所有视图使用相同的render resolution
        render_res = random.randint(self.cfg.render_image_res_low, self.cfg.render_image_res_high)
        region_size = self.cfg.render_region_size
        
        # 从预计算数据中提取选定视图
        selected_images = camera_tensors['images'][selected_indices]  # [num_views, 1, H_orig, W_orig]
        selected_opencv_poses = camera_tensors['opencv_poses'][selected_indices]  # [num_views, 4, 4]
        selected_intrinsics = camera_tensors['intrinsics'][selected_indices]  # [num_views, 3, 3]
        
        # Resize图像到render resolution
        if selected_images.shape[-1] != render_res or selected_images.shape[-2] != render_res:
            selected_images = F.interpolate(
                selected_images,
                size=(render_res, render_res),
                mode='bilinear',
                align_corners=False
            )
        
        # 随机裁剪
        anchors = torch.randint(
            0, render_res - region_size + 1, 
            size=(num_views, 2)
        )
        
        # 并行裁剪
        crop_indices = torch.arange(0, region_size, device=selected_images.device)
        index_i = (anchors[:, 0].unsqueeze(1) + crop_indices).view(-1, region_size, 1)
        index_j = (anchors[:, 1].unsqueeze(1) + crop_indices).view(-1, 1, region_size)
        batch_indices = torch.arange(num_views, device=selected_images.device).view(-1, 1, 1)
        
        cropped_images = selected_images[batch_indices, :, index_i, index_j]
        
        # 构建标准相机参数 [25维]
        standard_cameras = build_camera_standard(selected_opencv_poses, selected_intrinsics)
        
        # 准备返回数据
        resolutions = torch.full((num_views, 1), render_res, dtype=torch.int)
        
        return {
            "images": cropped_images,  # [num_views, 1, region_size, region_size]
            "cameras": standard_cameras,  # [num_views, 25] - 标准格式
            "anchors": anchors,
            "resolutions": resolutions,
        }


