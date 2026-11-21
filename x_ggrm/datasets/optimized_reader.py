import os
import h5py
import json
import numpy as np
import torch
from typing import Dict
from dataclasses import dataclass

@dataclass
class OptimizedSceneInfo:
    """优化数据格式的场景信息"""
    vol: np.ndarray
    all_camera_tensors: Dict[str, torch.Tensor]
    scanner_cfg: Dict
    metadata: Dict


class OptimizedDataReader:
    """优化数据格式的读取器"""
    
    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        self.hdf5_path = os.path.join(scene_dir, "optimized_data.h5")
        self.metadata_path = os.path.join(scene_dir, "metadata.json")
        
        # 验证文件存在
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # 加载元数据
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # 缓存原始投影形状
        self.original_proj_shape = self.metadata['original_projection_shape']

    def load_scene_info(self) -> OptimizedSceneInfo:
        """加载场景信息，返回统一的all_camera_tensors"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # 加载体积数据
            vol = f['vol_gt'][:]
            
            # 并行加载所有数据
            all_projections = torch.from_numpy(f['projections'][:]).float()
            all_rppc_params = torch.from_numpy(f['rppc_params'][:]).float()
            all_plucker_embeds = torch.from_numpy(f['plucker_embeds'][:]).float()
            
            # 加载相机参数
            all_camera_tensors = {}
            for key in f['camera_params'].keys():
                all_camera_tensors[key] = torch.from_numpy(f[f'camera_params/{key}'][:]).float()
            
            # 将投影数据和预计算参数添加到相机参数中
            all_camera_tensors['images'] = all_projections.unsqueeze(1)  # [N, 1, H, W]
            all_camera_tensors['rppc_params'] = all_rppc_params  # [N, H, W, 6] (无需resize)
            all_camera_tensors['plucker_embeds'] = all_plucker_embeds  # [N, 6]
        
        return OptimizedSceneInfo(
            vol=vol,
            all_camera_tensors=all_camera_tensors,
            scanner_cfg=self.metadata['scanner_cfg'],
            metadata=self.metadata
        )



def load_optimized_scene(scene_dir: str) -> OptimizedSceneInfo:
    """加载优化格式的场景数据"""
    reader = OptimizedDataReader(scene_dir)
    return reader.load_scene_info()

# 注册到reader回调中
def register_optimized_reader():
    """注册优化数据读取器到全局回调中"""
    try:
        from .reader import sceneLoadTypeCallbacks
        sceneLoadTypeCallbacks["OptimizedBlender"] = load_optimized_scene
    except ImportError:
        pass  # 如果reader模块不存在，忽略注册


# 自动注册
register_optimized_reader()