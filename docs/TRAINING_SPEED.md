# XLRM 单病例训练速度：E24

2026-11-21 在 GPU0（H100 PCIe 80 GB）、base_cy / ct、PyTorch 2.7.1+cu128 上实测。保持 LNDb-0045、24 张 224² 输入、每步 10,000 CT 点、368 维特征、全参数训练和 CT MSE。没有修改 E23 的模型权重或结果。

## 十分钟的主要来源

E23 追加 4,900 步至末次评价耗时 612.21 秒，平均约 125 ms/步。每步都重新执行图像编码、12 层图像 Transformer、12 层三平面 Transformer、局部 CNN、点查询及反向传播。24 张图像合计 4,704 个图像 tokens，三平面为 3×32²=3,072 个 tokens。368 维是末端密度 MLP 的输入，前端 Transformer 的规模保持原配置。

原实现还启用了三平面 Transformer 的 activation checkpointing，反向时重新计算前向激活，以节省显存。当前机器有足够显存，可以关闭重算。

原注意力已使用 FlashAttention。点查询前向只占约 2.7 ms/步。参考同步分段计时，三平面生成约 32.8 ms，局部 CNN 前向约 4.5 ms，整体反向约 67.3 ms，梯度裁剪与 AdamW 合计约 19.6 ms。

单独测得 128³ 全网格评价中位耗时 0.410 秒，完整 checkpoint（约 918 MB）的 torch.save 耗时 0.922 秒。保存计时不包含 fsync，与原训练器相同。按本轮 11 次评价、10 次保存粗估共约 14 秒，降低保存频率对总用时的收益较小。

## 同步性能测试

每组从相同模型、优化器和随机状态恢复，预热 5 步，计时 30 步。使用 CUDA events 分段计时，同时用同步后的 wall time 记录总时间。CPU 每步读取 loss 的行为保留。Profiler 另行运行，未混入该计时。

| 配置 | 单步 wall time / ms | 张量峰值显存 / GiB |
|---|---|---|
| 原流程 | 130.24 | 3.16 |
| 关闭激活重算 | 120.88 | 3.87 |
| fused AdamW | 119.92 | 3.16 |
| 关闭重算 + fused AdamW | **105.96** | **3.87** |
| 再启用 cuDNN 自动调优 | 106.30 | 3.87 |

组合提速约 1.23 倍，单步时间减少约 18.6%，张量峰值显存增加约 0.71 GiB。cuDNN 自动调优在此输入上没有收益，未启用。

## 同起点 100 步续训检查

两组均从 E23 第 5,000 步恢复，独立追加 100 步至 5,100 步。模型、AdamW、采样器和 PyTorch/CUDA 随机状态恢复，起点预测逐元素一致。沿用相同 128³ 留出网格与 [0,1] 评价范围。

| 配置 | 到末次评价耗时 / 秒 | 第 5,100 步 PSNR / dB |
|---|---|---|
| 原流程 | 15.254 | 28.0105 |
| 加速配置 | 12.307 | 28.0762 |

这段端到端短测提速约 1.24 倍，耗时减少约 19.3%；该计时包括起点评价和末次评价，不含之后的 checkpoint 写出和绘图。两组最终点生成器状态相同，实际优化器计数均为 5,100。加速配置确实使用 fused AdamW 并关闭 activation checkpointing。

GPU grid_sample / BF16 反向存在数值变动，原流程自身重复单步也不逐元素一致。本次加速单步前向与 loss 一致，更新后最大参数差约 6.2e-7，低于原流程单步重复的最大参数差约 1.82e-5。100 步短试没有观察到 PSNR 下降；不根据这一个小差值宣称精度改善。没有重新运行整段 5,000 步来测长期加速耗时。

两组 5,100 步 checkpoint 属于独立性能验证，未替换 E23 的 5,000 步 benchmark 权重。有一次启动短测时漏传 --resume，已明确终止并保存在 discarded_missing_resume/，排除于所有性能与精度对比。

## 当前默认配置

`configs/train-lrm-lndb-reprojection.yaml` 已设置：

```yaml
model:
  activation_checkpointing: false
train:
  optim:
    fused: true
```

单病例拟合入口和正式 LNDb 训练器均读取这些选项。旧配置没有这些字段时，继续使用 activation checkpointing=true、fused=false。模型参数结构和 checkpoint 参数名不变。显存受限时可显式将 activation_checkpointing 设为 true。

从旧 checkpoint 恢复时，优化器动量和计数继续恢复，backend 标记采用当前配置，避免旧 param_groups 将 fused 选项覆盖回去。

## 复测与证据

宿主机结果根目录为 `/data/wangcy/experiment/sasplat/results/XLRM/lndb0045_speed_profile_20261121/`。最终数据为 `final/benchmark.json`、`final/profile.txt`、`io_profile.json`、`quality_speed_check.json`，短程训练为 `control5100/` 和 `fast5100/`，正式训练器验证为 `native_smoke/smoke.json`。

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 -e TORCHDYNAMO_DISABLE=1 \
  -e PYTHONPATH=/workspace/recon/X_videoDiffusion -e OMP_NUM_THREADS=4 \
  -w /workspace/recon/X_videoDiffusion base_cy /opt/conda/envs/ct/bin/python -u \
  -m tools.benchmark_single_case_speed \
  --checkpoint /data/experiment/sasplat/results/XLRM/lndb0045_reprojection368_resume5000_20261121/step5000.pth \
  --dataset-root /data/experiment/sasplat/lndb \
  --output /data/experiment/sasplat/results/XLRM/lndb0045_speed_profile_repeat \
  --warmup 5 --steps 30
```

进一步减少 Transformer 层数或图像 tokens 会改变模型容量，需要另做重建质量对照。增加监督点批量会改变每步预算，应同时报告总监督点数和优化步骤。

## 流程参考

性能分析流程参考 Scientific Agent Skills 的 optimize-for-gpu skill，数值与结论来自上述服务器实测。

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
