# XLRM 几何对齐的多尺度投影特征分支

当前配置将三平面点特征的 144 维，与四尺度投影特征的 224 维拼接，使用 368→128→128→128→1 的密度 MLP。三个平面各 48 通道，四个二维尺度为 [16,32,64,112]，直接拼接，feature_projection 为 Identity。

本地 DeepSparse LNDb 配置为 128 维三维特征加 16+32+64+128 维二维特征，共 368 维。XLRM 保留三个等通道平面，总维度同为 368，全部通道数为 8 的倍数。此前 E21 的 624 维配置和结果保存在原实验目录。训练使用 CT 点 MSE，两个分支和解码器一起更新。

```mermaid
flowchart LR
    I[输入投影与射线编码] --> L[原 XLRM 编码器与 Transformer]
    L --> T[三个 64² 三平面]
    T --> S[在三维点采样 144 维]
    I --> C[投影强度 CNN]
    C --> F[四尺度二维特征图]
    P[三维查询点与输入相机] --> U[计算探测器坐标]
    U --> Q[双线性采样与跨视角 max]
    F --> Q
    Q --> V[224 维局部特征]
    S --> M[拼接 368 维后输入密度 MLP]
    V --> M
    M --> D[CT 密度]
```

## 特征与几何

- 新 CNN 独立处理每张投影，只读取原七通道输入的强度通道，保持原强度尺度。224² 输入对应的特征图为 16×224²、32×112²、64×56²、112×28²。
- 每个三维点根据同一张输入图的相机投到探测器，四个尺度分别双线性采样，跨有效视角取 max，再按尺度拼接。相机后方、探测器外和被局部分支 mask 禁用的视角不参与 max，全部无效时局部特征为零。
- 相机是现有 XLRM 的 25 个数：按行展开的 Blender `world_from_camera` 4×4，再接图像归一化的 OpenCV 内参 3×3。投影时转到 OpenCV 相机坐标。三维点与相机平移均使用 XLRM 世界单位，LNDb 中 200 mm 立方体映射到 `[-1,1]³`。
- 新分支使用常规 `(column,row)` 网格、`grid_sample(align_corners=False)` 和像素中心内参。DeepSparse 本地代码使用转置特征图及 `align_corners=True`；这里沿用 XLRM 的像素中心约定，并以原射线采样器的往返测试验证。
- 几何在 FP32 中计算，三维矩阵乘法用显式分量求和，避免本机 TF32 设置降低坐标精度。对应修改也用于原 RaySampler 的方向旋转。
- 网络输入只有投影和相机。CT 只提供损失标签。`encode()` 返回当前病例的特征与相机，评价时由调用方复用，模型不缓存跨病例的投影坐标。

这借鉴了 DeepSparse 查询二维局部特征的机制。新增模块使用独立的轻量 CNN，XLRM 的三平面路径继续提供全局特征。

## 接口和配置

`configs/train-lrm-lndb-reprojection.yaml` 开启 `model.reprojection_channels: [16,32,64,112]`，三平面各 48 通道，`point_feature_dim: null`，每例 10,000 CT 点，像素和 TV 损失为零。多患者训练配置为 200 轮，验证使用重采样 128³ CT，患者划分沿用原数据。

原模型继续使用原配置和 `reprojection_channels=None`，已有参数名和形状保持兼容。当前 368 维配置改变了上采样器通道和 MLP 首层，需使用对应的新 checkpoint；当前入口不会自动改写旧权重。复现 E21 的 624 维模型时应使用该实验目录保存的 config.yaml。

单病例脚本在 `--reprojection` 模式下读取配置中的三平面通道与二维通道。`--direct-features` 表示直接拼接三平面特征，不增加线性投影。历史非重投影拟合默认仍为每面 128 通道，也可通过 `--triplane-channels` 显式指定。

```python
# image: [B,V,H,W,7]; input_cameras: [B,V,25]; points: [B,P,3]
output = model(image, input_cameras=input_cameras, ct_query_points=points)

# 评价时只编码一次，再分块查询。
context = model.encode(image, input_cameras)
density = model.query_points(context, points)['sigma']
```

开启分支后使用上述入口。直接调用 `forward_planes()` 只能得到三平面，不能提供二维局部特征。当前分支的训练与评价入口支持 CT 点和 CT 体积，监督保持 CT MSE。

## 当前 368 维配置的验证

2026-11-21 完成几何、梯度、实际通道和旧配置兼容检查。四个二维特征图的最后一级为 [1,24,112,28,28]，点 MLP 首层输入为 [10000,368]。两条路径梯度均非零。使用相同 LNDb-0045、24 帧、10,000 CT 点/步和 128³ 偶数原生坐标留出网格，从随机初始化训练 100 步。

| 配置 | 三平面维数 | 二维局部维数 | 合计 | 100 步 PSNR / dB |
|---|---|---|---|---|
| E21 旧配置 | 384 | 240 | 624 | 21.1301 |
| E22 当前配置 | 144 | 224 | 368 | 21.0536 |
| 本地 DeepSparse 配置 | 128 | 240 | 368 | 本轮未训练 |

总输入维度减少 41.03%，短程 PSNR 差为 -0.0765 dB。三平面上游、MLP 隐层、CT MSE 和优化设置保持原设置。不同架构的初始化不同，本次结果描述同一病例的早期拟合，长期效果待验证。

当前证据位于 `/data/wangcy/experiment/sasplat/results/XLRM/lndb0045_reprojection368_check_20261121/checks.json` 和 `lndb0045_reprojection368_scratch100_20261121/`，后者包含 config.yaml、metrics.csv、summary.json、step100.pth 和 step100.npy。旧 624 维输出保持原样。

## E21 原 624 维配置的历史验证

2026-11-21，容器 `base_cy`、环境 `ct`、GPU0：

- LNDb-0045 的矩阵投影与独立圆轨迹公式的最大归一化误差为 `5.96e-7`。
- 原射线采样器的非中心像素往返误差为 `2.38e-7`，在 224² 探测器约为 `0.000027` 像素。AMP 开关不改变几何结果。
- 像素坐标坡度图采样、全不可见点、视角同步置换、局部分支 mask、输入强度变化、分块查询和网格/点查询一致性检查通过。
- FP32 分块查询最大差为 `2.38e-7`。BF16 不同查询批量下最大差为 `0.00390625`，对应输出量化，单独记录。
- 新二维编码器、三平面上采样器和 MLP 新增输入列均有非零有限梯度。关闭分支后，旧第 5,000 步 checkpoint 严格加载，原点查询与新模型入口的预测逐元素一致。
- 正式 LNDb 训练器的单步训练及一次 32³ 验证通过，训练标签来自 LNDb-0007 的原生 CT，`projection_mse=null`。

LNDb-0045、24 视角的配对短程测试使用相同的 10,000 点/步、种子、CT MSE、AdamW 和 128³ 偶数原生坐标留出网格。两组均随机初始化，全参数训练；新增分支改变了 MLP 首层结构与初始化。预测裁剪到 `[0,1]`，固定强度范围 1。

| 模型 | 起点 PSNR / dB | 50 步 PSNR / dB | 100 步 PSNR / dB | 100 步 3D SSIM |
|---|---|---|---|---|
| 原 384 维 XLRM | 6.7192 | 12.7827 | 17.4510 | 0.3467 |
| 384+240 维重投影分支（旧配置） | 7.0893 | 12.0433 | 21.1301 | 0.5350 |

100 步差值为 +3.6791 dB。这是单病例早期拟合结果。新增模型的 5,000 步成绩和多患者泛化待运行。

另外从新增模型第 100 步完整状态恢复，单独追加 1 步验证恢复入口，累计第 101 步为 21.1666 dB。恢复起点逐体素预测一致。此诊断保存在独立目录，未并入上表的 100 步成绩。

宿主机证据目录位于 `/data/wangcy/experiment/sasplat/results/XLRM/`：

- `lndb0045_reprojection_check_20261121/`：`checks.json`、`fit_comparison.json`、`comparison_100.png`、`trainer_smoke/smoke.json` 及各验证日志。
- `lndb0045_reprojection_scratch100_20261121/`：新增分支的 100 步实验。
- `lndb0045_reprojection_control100_20261121/`：关闭分支的 100 步配对对照。
- `lndb0045_reprojection_resume101_check_20261121/`：累计 101 步的恢复诊断。

## 服务器命令

从宿主机执行几何与梯度检查：

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 -e TORCHDYNAMO_DISABLE=1 \
  -e PYTHONPATH=/workspace/recon/X_videoDiffusion -e OMP_NUM_THREADS=4 \
  -w /workspace/recon/X_videoDiffusion base_cy /opt/conda/envs/ct/bin/python \
  -m tools.check_reprojection \
  --config configs/train-lrm-lndb-reprojection.yaml \
  --dataset-root /data/experiment/sasplat/lndb \
  --legacy-checkpoint /data/experiment/sasplat/results/single_case/lndb0045_24views_resume5000_20261121/xlrm384/step5000.pth \
  --output /data/experiment/sasplat/results/XLRM/lndb0045_reprojection368_check_20261121/checks.json
```

以下命令可从已保存的第 100 步继续至累计 5,000 步，每 500 步评价；尚未执行这次长期训练。输出目录需不存在。

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 -e TORCHDYNAMO_DISABLE=1 \
  -e PYTHONPATH=/workspace/recon/X_videoDiffusion -e OMP_NUM_THREADS=4 \
  -w /workspace/recon/X_videoDiffusion base_cy /opt/conda/envs/ct/bin/python -u \
  -m tools.fit_wide_point_features \
  --config configs/train-lrm-lndb-reprojection.yaml \
  --dataset-root /data/experiment/sasplat/lndb \
  --e13-root /data/experiment/sasplat/results/single_case/lndb0045_24views_scratch_1000_20261120 \
  --output /data/experiment/sasplat/results/XLRM/lndb0045_reprojection368_resume5000_20261121 \
  --direct-features --reprojection --steps 5000 --eval-every 500 \
  --resume /data/experiment/sasplat/results/XLRM/lndb0045_reprojection368_scratch100_20261121/step100.pth
```

宿主机 `/data/wangcy` 挂载为容器 `/data`，`/home/wangcy/projects` 挂载为 `/workspace`。普通多患者训练可在新的输出工作目录中运行 `python -m x_ggrm.runners.train.lndb_lrm --config /workspace/recon/X_videoDiffusion/configs/train-lrm-lndb-reprojection.yaml`；以上两类完整训练均尚未执行。
