# LNDb-0045 固定 24 视角测试样本

本样本用于 X-LRM 输入检查、单病例拟合和重建调试。病例来自现有 LNDb 训练划分，沿用探索实验 E12、E13 的 24 帧。投影来自已有的 CT 仿真采集，导出保留存档像素和对应角度。病例归属应在后续评价中保留。

## 文件

| 文件 | 内容 |
|---|---|
| `volume.npz` | `ct_xyz`，uint8，完整 256×256×256 CT，轴序 XYZ |
| `projections.npz` | `projections`，float32，24×512×512 原始投影，含存档负值；`angles_rad`；`source_frame_indices` |
| `input_views.npz` | `input_views`，float32，24×224×224×7，与现有 XLRM 数据 adapter 的输出逐元素一致 |
| `detector_png/` | 24 张原分辨率 512² 的 uint16 PNG 投影图 |
| `input_png/` | 24 张实际输入分辨率 224² 的 uint16 PNG 图像通道 |
| `projection_montage.png` | 按帧顺序排列的 24 张输入图概览 |
| `acquisition.yaml` | 原采集配置，包含空间、设备、角度和噪声设置 |
| `geometry.json` | 每视角源点、探测器中心与方向、相机正逆矩阵、内参、轴序和强度约定 |
| `frames.csv` | 包内帧索引、源帧索引、弧度、角度及 PNG 对应关系 |
| `metadata.json` | 数据形状、选帧规则、PNG 强度解码比例和导出检查结果 |

源投影共有 50 帧，选取 `floor(linspace(0, 50, 24, endpoint=False))`，索引为 `[0,2,4,6,8,10,12,14,16,18,20,22,25,27,29,31,33,35,37,39,41,43,45,47]`。本包覆盖角度为 0°～169.2°，原 50 帧轨迹为 0°～176.4°。

## 加载

在仓库根目录运行下面的 Python 代码。

```python
from pathlib import Path
import json
import numpy as np
import torch

root = Path('examples/LNDb-0045')
with np.load(root / 'volume.npz', allow_pickle=False) as data:
    ct = data['ct_xyz'].astype(np.float32) / 255
with np.load(root / 'projections.npz', allow_pickle=False) as data:
    projections = data['projections']
    angles = data['angles_rad']
    source_frames = data['source_frame_indices']
with np.load(root / 'input_views.npz', allow_pickle=False) as data:
    input_views = torch.from_numpy(data['input_views'])[None]
geometry = json.loads((root / 'geometry.json').read_text())

assert ct.shape == (256, 256, 256)
assert projections.shape == (24, 512, 512)
assert input_views.shape == (1, 24, 224, 224, 7)

# model 为按目标 checkpoint 配置构建并加载的 ModelLRM。
# planes = model.forward_planes(input_views.to(next(model.parameters()).device))
# prediction = model.synthesizer.forward_grid(planes, grid_size=256)['sigma']
```

七个通道依次为图像线积分、源点 XYZ、单位射线方向 XYZ。源点位于 XLRM 模型坐标，射线方向使用 CT 世界轴。图像通道由 `max(raw_projection, 0) × 10` 得到，再通过 PyTorch bilinear、`align_corners=False`、`antialias=True` 缩放到 224²。`input_views.npz` 已完成这些处理，可以直接送入模型。

PNG 以所有帧共享的强度窗口保存，解码到模型线积分时将像素值乘以 `metadata.json` 中的 `png_decode_model_integral_scale`。PNG 编码会量化并截断存档负值。定量实验使用 NPZ 内的浮点数组。

## 几何与 CT 坐标

CT 物理边长为 200 mm，原生体素间距为 0.78125 mm。源到等中心为 500 mm，源到探测器为 700 mm。探测器边长为 400 mm，原生像素间距为 0.78125 mm。

`geometry.json` 中带 `_m` 后缀的矩阵及位置采用米，模型长度为每米 10 个单位。`world_from_camera` 将相机坐标转换到以 CT 等中心为原点的世界坐标。Blender 相机沿 −Z 观察、+Y 朝上，OpenCV 相机沿 +Z 观察、+Y 朝下。两套相机正逆矩阵逐视角保存，探测器 U 对应列增加，V 对应行增加。

NPZ 的 CT 轴序直接沿用原数据 adapter 的 XYZ 数组。原 NIfTI 头使用索引空间，实际物理尺寸来自 `acquisition.yaml`。采集体素中心按 `(i+0.5) × extent/N − extent/2` 定义。现有 XLRM 点查询沿用 `q=2i/(N−1)−1`，两种坐标约定均在 `geometry.json` 记录，便于复现当前实现和检查边界。

`acquisition.yaml` 记录了原数据生成时的噪声配置。本包导出复用了存档投影，导出步骤没有额外加入像素噪声或几何扰动。

## 复现导出

在仓库根目录使用现有预处理 LNDb 数据运行，输出目录需要尚未创建。

```bash
python -m tools.export_example_case \
  --dataset-root /path/to/preprocessed/lndb \
  --case LNDb-0045 --views 24 --output /path/to/new-output-directory
```

导出检查包括 CT、投影、七通道输入的精确回读，合法旋转及正逆矩阵一致性，以及探测器角点与中心附近像素的几何射线同输入方向的一致性。保存的最大方向分量差为 `8.78e-8`，最大相机正逆矩阵误差为 `2.23e-16`。

## 来源与授权

来源为 LNDb 预处理病例。仓库所有者于 2026-11-21 确认具有额外公开授权，并授权发布本包。使用及进一步分发时请核对相应授权和 LNDb 数据条款。数据来源及署名见 [LNDb 官方规则](https://lndb.grand-challenge.org/Rules/) 和 [LNDb 数据集论文](https://arxiv.org/abs/1911.08434)。
