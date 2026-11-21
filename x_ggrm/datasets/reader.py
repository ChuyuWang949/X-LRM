import sys
from typing import NamedTuple
import numpy as np
import os.path as osp
import json
import torch

sys.path.append("./")

mode_id = {
    "parallel": 0,
    "cone": 1,
}

class CameraInfo_Neural_Field(NamedTuple):
    uid: int
    c2w: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int
    image: np.array

class CameraInfo_Gaussian_Splatting(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    c2w: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    width: int
    height: int
    mode: int


class SceneInfo_Neural_Field(NamedTuple):
    train_cameras: list
    test_cameras: list
    vol: torch.tensor

class SceneInfo_Gaussian_Splatting(NamedTuple):
    train_cameras: list
    test_cameras: list
    vol: torch.tensor
    scanner_cfg: dict


def read_info_neural_field(path, eval):
    """Read blender format CT data."""
    # Read meta data
    meta_data_path = osp.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    meta_data["vol"] = osp.join(path, meta_data["vol"])

    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(
            np.array(meta_data["scanner"]["sVoxel"])
            / np.array(meta_data["scanner"]["nVoxel"])
        )
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(
            np.array(meta_data["scanner"]["sDetector"])
            / np.array(meta_data["scanner"]["nDetector"])
        )

    #! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        meta_data["scanner"][key_to_scale] = (
            np.array(meta_data["scanner"][key_to_scale]) * scene_scale
        ).tolist()

    cam_infos = read_cameras_neural_field(meta_data, path, eval)
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]

    vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float()
    scene_info = SceneInfo_Neural_Field(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        vol=vol_gt
    )
    return scene_info


def read_cameras_neural_field(meta_data, source_path, eval=False):
    """Read camera info."""
    cam_cfg = meta_data["scanner"]

    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]

    cam_infos = {"train": [], "test": []}
    for split in splits:
        split_info = meta_data["proj_" + split]
        n_split = len(split_info)
        if split == "test":
            uid_offset = len(meta_data["proj_train"])
        else:
            uid_offset = 0
        for i_split in range(n_split):
            frame_info = meta_data["proj_" + split][i_split]
            frame_angle = frame_info["angle"]

            # CT 'transform_matrix' is a camera-to-world transform
            DSO = cam_cfg["DSO"]

            c2w = angle2pose_opencv(DSO, frame_angle)  # c2w

            image_path = osp.join(source_path, frame_info["file_path"])
            image = np.load(image_path)
            # Note, dDetector is [v, u] not [u, v]
            FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
            FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2

            cam_info = CameraInfo_Neural_Field(
                uid=i_split + uid_offset,
                c2w=c2w,
                FovY=FovY,
                FovX=FovX,
                width=cam_cfg["nDetector"][1],
                height=cam_cfg["nDetector"][0],
                image=image,
            )

            cam_infos[split].append(cam_info)

    return cam_infos

def angle2pose_opencv(DSO, angle):
    """
    Generate camera pose for CT scanner that can properly sample [-1,1]³ volume.
    Key fix: Camera looks toward origin with proper Z-axis alignment.
    """
    # 1) Camera position on circular orbit at z=0
    trans = np.array([DSO*np.cos(angle), DSO*np.sin(angle), 0.], dtype=np.float32)

    # 2) Camera coordinate system: looking TOWARD origin (not away)
    # Forward direction (camera -Z) points toward origin  
    forward = -trans / np.linalg.norm(trans)  # Normalized vector TO origin
    
    # 3) Build orthonormal basis
    world_up = np.array([0., 0., 1.], dtype=np.float32)  # World Z-up
    
    # Right direction (camera +X)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    
    # Up direction (camera +Y) 
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    
    # 4) Build camera-to-world rotation matrix
    # OpenCV convention: [right, up, -forward] (camera +Z points away from target)
    R = np.column_stack([right, up, -forward])
    
    # 5) Assemble 4x4 transform
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = trans
    
    return c2w



def read_info_gaussian_splatting(path, eval):
    """Read blender format CT data."""
    # Read meta data
    meta_data_path = osp.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    meta_data["vol"] = osp.join(path, meta_data["vol"])

    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(
            np.array(meta_data["scanner"]["sVoxel"])
            / np.array(meta_data["scanner"]["nVoxel"])
        )
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(
            np.array(meta_data["scanner"]["sDetector"])
            / np.array(meta_data["scanner"]["nDetector"])
        )

    #! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        meta_data["scanner"][key_to_scale] = (
            np.array(meta_data["scanner"][key_to_scale]) * scene_scale
        ).tolist()

    cam_infos = read_cameras_gaussian_splatting(meta_data, path, eval, scene_scale)
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]

    vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float()

    scene_info = SceneInfo_Gaussian_Splatting(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        vol=vol_gt,
        scanner_cfg=meta_data["scanner"]
    )
    return scene_info


def read_cameras_gaussian_splatting(meta_data, source_path, eval=False, scene_scale=1.0):
    """Read camera info."""
    cam_cfg = meta_data["scanner"]

    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]

    cam_infos = {"train": [], "test": []}
    for split in splits:
        split_info = meta_data["proj_" + split]
        n_split = len(split_info)
        if split == "test":
            uid_offset = len(meta_data["proj_train"])
        else:
            uid_offset = 0
        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_info = meta_data["proj_" + split][i_split]
            frame_angle = frame_info["angle"]

            # CT 'transform_matrix' is a camera-to-world transform
            c2w = angle2pose_blender(cam_cfg["DSO"], frame_angle)  # c2w
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = osp.join(source_path, frame_info["file_path"])
            image = np.load(image_path) * scene_scale
            # Note, dDetector is [v, u] not [u, v]
            FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
            FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2

            mode = mode_id[cam_cfg["mode"]]

            cam_info = CameraInfo_Gaussian_Splatting(
                uid=i_split + uid_offset,
                R=R,
                T=T,
                c2w=c2w,
                FovY=FovY,
                FovX=FovX,
                image=image,
                width=cam_cfg["nDetector"][1],
                height=cam_cfg["nDetector"][0],
                mode=mode,
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")
    return cam_infos


def angle2pose_blender(DSO, angle):
    """Transfer angle to pose (c2w) based on scanner geometry.
    1. rotate -90 degree around x-axis (fixed axis),
    2. rotate 90 degree around z-axis  (fixed axis),
    3. rotate angle degree around z axis  (fixed axis)"""

    phi1 = -np.pi / 2
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ]
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R3 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return transform

def read_info_grm_optimized(path, eval):
    """
    Optimized GRM data reader that returns tensors directly, avoiding Camera object creation.
    Specifically designed for tensor-based GRM training pipeline.
    """
    # Read meta data
    meta_data_path = osp.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    meta_data["vol"] = osp.join(path, meta_data["vol"])

    # Ensure all required scanner parameters exist
    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(
            np.array(meta_data["scanner"]["sVoxel"])
            / np.array(meta_data["scanner"]["nVoxel"])
        )
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(
            np.array(meta_data["scanner"]["sDetector"])
            / np.array(meta_data["scanner"]["nDetector"])
        )

    # Scene scaling to normalize volume to [-1, 1]^3
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    
    # Scale all relevant parameters
    for key_to_scale in [
        "dVoxel", "sVoxel", "sDetector", "dDetector",
        "offOrigin", "offDetector", "DSD", "DSO",
    ]:
        meta_data["scanner"][key_to_scale] = (
            np.array(meta_data["scanner"][key_to_scale]) * scene_scale
        ).tolist()

    # Read camera tensors directly
    camera_tensors = read_camera_tensors_grm(meta_data, path, eval, scene_scale)
    
    # Load volume ground truth
    vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float()

    # Return simplified scene info with tensors
    return {
        'train_camera_tensors': camera_tensors["train"],
        'test_camera_tensors': camera_tensors["test"], 
        'vol': vol_gt,
        'scanner_cfg': meta_data["scanner"]
    }


def read_camera_tensors_grm(meta_data, source_path, eval=False, scene_scale=1.0):
    """
    Read camera parameters and return as batched tensors for GRM training.
    Optimized version that skips CameraInfo and Camera object creation.
    """
    cam_cfg = meta_data["scanner"]
    
    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]

    camera_tensors = {"train": {}, "test": {}}
    
    for split in splits:
        split_info = meta_data["proj_" + split]
        n_cameras = len(split_info)
        
        # Pre-allocate tensor lists
        projection_matrices = []
        world_view_transforms = []
        full_proj_transforms = []
        camera_centers = []
        images = []
        image_widths = []
        image_heights = []
        FovX_list = []
        FovY_list = []
        
        print(f"Reading {n_cameras} cameras for {split} split...")
        
        for i in range(n_cameras):
            frame_info = split_info[i]
            frame_angle = frame_info["angle"]

            # Compute camera transforms
            c2w = angle2pose_blender(cam_cfg["DSO"], frame_angle)
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # Transposed for CUDA compatibility
            T = w2c[:3, 3]

            # Load and scale image
            image_path = osp.join(source_path, frame_info["file_path"])
            image = np.load(image_path) * scene_scale
            
            # Convert to tensor format
            image_tensor = torch.from_numpy(image).float()
            if len(image_tensor.shape) == 2:  # Add channel dimension if grayscale
                image_tensor = image_tensor.unsqueeze(0)  # [1, H, W]
            
            # Camera parameters
            FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
            FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2
            width = cam_cfg["nDetector"][1]
            height = cam_cfg["nDetector"][0]
            
            # Compute projection matrix (K[R|t])
            fx = fy = width / (2 * np.tan(FovX * 0.5))
            cx = width / 2.0
            cy = height / 2.0
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            Rt = w2c[:3, :]  # [3, 4]
            P = np.matmul(K, Rt)  # [3, 4]
            
            # Build full projection transform (4x4)
            full_proj = np.zeros((4, 4), dtype=np.float32)
            full_proj[:3, :] = P
            full_proj[3, 3] = 1.0
            
            # Camera center in world coordinates
            camera_center = c2w[:3, 3]
            
            # Convert to tensors and append
            projection_matrices.append(torch.from_numpy(P).float())
            world_view_transforms.append(torch.from_numpy(w2c).float())
            full_proj_transforms.append(torch.from_numpy(full_proj).float())
            camera_centers.append(torch.from_numpy(camera_center).float())
            images.append(image_tensor)
            image_widths.append(float(width))
            image_heights.append(float(height))
            FovX_list.append(float(FovX))
            FovY_list.append(float(FovY))

        # Stack into batched tensors
        camera_tensors[split] = {
            'projection_matrices': torch.stack(projection_matrices),  # [N, 3, 4]
            'world_view_transforms': torch.stack(world_view_transforms),  # [N, 4, 4]
            'full_proj_transforms': torch.stack(full_proj_transforms),  # [N, 4, 4]
            'camera_centers': torch.stack(camera_centers),  # [N, 3]
            'images': torch.stack(images),  # [N, C, H, W]
            'image_widths': torch.tensor(image_widths, dtype=torch.float32),  # [N]
            'image_heights': torch.tensor(image_heights, dtype=torch.float32),  # [N]
            'FovX': torch.tensor(FovX_list, dtype=torch.float32),  # [N]
            'FovY': torch.tensor(FovY_list, dtype=torch.float32),  # [N]
        }
        
        print(f"Loaded {n_cameras} cameras for {split} split as tensors")
    
    return camera_tensors


sceneLoadTypeCallbacks = {
    "Blender_neural_field": read_info_neural_field,
    "Blender_gaussian_splatting": read_info_gaussian_splatting,
    "Blender_grm": read_info_grm_optimized,  # New optimized GRM reader
}
