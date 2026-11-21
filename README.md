# X-LRM

X-LRM reconstructs a 3D CT volume from sparse CBCT projections. It encodes each projection and its ray geometry as patch tokens, fuses tokens across views, and uses a triplane representation and a point decoder to predict attenuation.

This repository contains the model, LNDb dataset adapter, training configurations, and volume export tool.

Exploratory LNDb experiments, including training budgets, initialization, evaluation protocols, and completed or pending work, are recorded in [实验记录](docs/EXPERIMENTS.md).

A fixed [LNDb-0045 example](examples/LNDb-0045/README.md) includes 24 input views, a reference CT volume, projection images, and per-view camera geometry for reconstruction debugging.

## Environment

The LNDb pipeline was checked with Python 3.11, PyTorch 2.7.1, and a CUDA GPU.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## LNDb data

Set `dataset.lndb_root` to your preprocessed dataset. The adapter expects the following layout.

```text
data/lndb/
├── config.yaml
├── splits.json
├── processed/<case>.nii.gz
├── projections/<case>/all.pickle
└── blocks/
    ├── blocks.npy
    └── <case>/block_<index>.npy
```

`splits.json` holds disjoint `train`, `eval`, and `test` case lists. The geometry is centered and uses metres, and projection angles are in radians. The CT arrays are uint8 in the adapter's XYZ order. Projection pickles provide `projections` and `angles`. Use pickles produced by a trusted preprocessing pipeline. The `blocks` directory is required for CT point supervision.

## Training and inference

Train a 24-view model with CT point supervision from the repository root.

```bash
python -m x_ggrm.runners.train.lndb_lrm \
  --config configs/train-lrm-lndb-distill.yaml \
  experiment.child=lndb24_ct dataset.lndb_root=./data/lndb \
  dataset.nums_input_view=24 distillation.weight=0 \
  train.ct_points_per_case=10000 train.loss.pixel_weight=0 train.loss.tv_weight=0
```

For 10-view CT-only training, use `dataset.nums_input_view=10` and a separate `experiment.child`. To enable dense-to-sparse triplane distillation, set `distillation.weight=0.1` and `distillation.teacher_checkpoint` to a trained 24-view checkpoint. Add `--smoke` for one training update and a validation forward.

Export reconstructions to XYZ NumPy volumes using a checkpoint that matches the model configuration.

```bash
python -m tools.export_xlrm_lndb \
  --config configs/train-lrm-lndb-distill.yaml \
  --checkpoint /path/to/model.safetensors --output ./predictions --views 10 --split test
```

The exporter reads `dataset.lndb_root` from the YAML file. Edit that field or provide your own config when using another data location.

## Third-party code

Parts of the model are adapted from [OpenLRM](https://github.com/3DTopia/OpenLRM) and [DINOv2](https://github.com/facebookresearch/dinov2). Their Apache 2.0 terms are in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). The marked EG3D-derived renderer files use the [NVIDIA EG3D license](LICENSES/NVIDIA-EG3D.txt), which limits use to noncommercial research or evaluation. The MIT notice for `math_utils.py` is retained in that file.
