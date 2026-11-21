#!/usr/bin/env python3
"""Export XLRM test volumes from shared LNDb projections."""
import argparse
import json
from pathlib import Path

import numpy as np
from accelerate import PartialState
from omegaconf import OmegaConf
from safetensors.torch import load_file
import torch

from x_ggrm.datasets.lndb_lrm import LNDbLRMDataset
from x_ggrm.models import ModelLRM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--views', nargs='+', type=int, required=True)
    parser.add_argument('--split', choices=('eval', 'test'), default='test')
    args = parser.parse_args()
    torch.set_num_threads(4)
    PartialState()
    cfg = OmegaConf.load(args.config)
    model = ModelLRM(**cfg.model, image_res=cfg.dataset.input_image_res)
    model.load_state_dict(load_file(args.checkpoint), strict=True)
    model.eval().cuda()

    for views in args.views:
        cfg.dataset.nums_input_view = views
        dataset = LNDbLRMDataset(cfg, args.split)
        indices = np.linspace(0, dataset.geometry['numTrain'], views, endpoint=False).astype(int)
        output = args.output / f'views{views}'
        output.mkdir(parents=True)
        (output / 'provenance.json').write_text(json.dumps({
            'checkpoint': str(args.checkpoint), 'config': str(args.config),
            'split': args.split, 'views': views, 'view_indices': indices.tolist(),
            'axes': 'xyz', 'inference_precision': cfg.train.mixed_precision,
        }, indent=2) + '\n')

        for index, case in enumerate(dataset.names):
            projections, _, _ = dataset.projection_inputs(index)
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=cfg.train.mixed_precision == 'bf16'):
                planes = model.forward_planes(projections[None].cuda())
                volume = model.synthesizer.forward_grid(planes, grid_size=model.volume_grid_size)['sigma']
            volume = volume[0, ..., 0].float().cpu().numpy()
            if volume.shape != (model.volume_grid_size,) * 3 or not np.isfinite(volume).all():
                raise ValueError(f'Invalid volume for {case}')
            np.save(output / f'{case}.npy', volume, allow_pickle=False)
            print(f'[{index+1}/{len(dataset)}] {views} views {case}: {volume.min():.4f}–{volume.max():.4f}',
                  flush=True)


if __name__ == '__main__':
    main()
