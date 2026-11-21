"""Fit a random-start XLRM with 368-dimensional point features to E13's case."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from accelerate import PartialState
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import SimpleITK as sitk
import torch
from torch.nn import functional as F

from x_ggrm.datasets.lndb_lrm import LNDbLRMDataset
from x_ggrm.models import ModelLRM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--e13-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    torch.set_num_threads(4)
    PartialState()
    torch.manual_seed(42)
    cfg = OmegaConf.load(args.config)
    cfg.dataset.lndb_root = str(args.dataset_root)
    cfg.model.triplane_dim = 128
    cfg.model.point_feature_dim = 368
    cfg.model.point_decoder_hidden_dim = 128
    cfg.saver.load_model = None
    cfg.train.freeze_point_decoder = False
    cfg.distillation = dict(weight=0.)
    cfg.train.loss.pixel_weight = 0.
    cfg.train.loss.tv_weight = 0.
    cfg.train.loss.vol_mse_weight = 1.
    OmegaConf.save(cfg, out / 'config.yaml')
    prior = json.loads((args.e13_root / 'xlrm/summary.json').read_text())
    case = prior['case']
    ds = LNDbLRMDataset(cfg, 'train')
    record = ds._read_projections(ds.names.index(case))
    inputs = ds._encode_projections(record, np.asarray(prior['views']))[0][None].cuda()
    ct = (torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(
        str(args.dataset_root / 'processed' / f'{case}.nii.gz')))).float() / 255).cuda()
    reference = ct[::2, ::2, ::2].contiguous().cpu().numpy()
    np.testing.assert_array_equal(reference, np.load(args.e13_root / 'heldout_ct.npy', allow_pickle=False))
    np.save(out / 'heldout_ct.npy', reference)
    axis = torch.arange(0, 256, 2, device='cuda')
    query = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(1, -1, 3).float() / 255 * 2 - 1
    rng = torch.Generator(device='cuda').manual_seed(prior['point_seed'])
    model = ModelLRM(**cfg.model, image_res=cfg.dataset.input_image_res).cuda()
    assert model.synthesizer.decoder.feature_projection.out_features == 368
    assert model.synthesizer.decoder.net[0].in_features == 368
    assert all(p.requires_grad for p in model.parameters())
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in parameters if p.ndim >= 2], weight_decay=.02),
        dict(params=[p for p in parameters if p.ndim < 2], weight_decay=0.)],
        lr=5e-5, betas=(.9, .95))
    rows = []
    losses = []
    point_hash = hashlib.sha256()
    stamp = time.monotonic()
    shape_checks = {}

    @torch.no_grad()
    def measure(step):
        model.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            planes = model.forward_planes(inputs)
            assert planes.shape == (1, 3, 128, 64, 64)
            predictions = torch.cat([model.synthesizer.forward_points(planes, q)['sigma'].reshape(-1).float().cpu()
                                     for q in query.split(131072, dim=1)]).numpy().reshape(reference.shape)
        mse = float(np.mean((np.clip(predictions, 0, 1).astype(np.float64) - reference.astype(np.float64)) ** 2))
        row = dict(step=step, psnr=float(-10 * np.log10(mse)), mse=mse,
                   train_mse=float(np.mean(losses[-50:])) if losses else None,
                   seconds=time.monotonic() - stamp)
        rows.append(row)
        with (out / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        np.save(out / f'step{step}.npy', predictions)
        print(json.dumps(row), flush=True)

    measure(0)
    for step in range(1, 1001):
        model.train()
        xyz = torch.randint(256, (12000, 3), device='cuda', generator=rng)
        xyz = xyz[(xyz % 2 == 1).any(dim=1)][:10000]
        assert xyz.shape == (10000, 3)
        if step in (1, 100, 300, 600, 1000):
            point_hash.update(xyz.cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        handles = []
        if step == 1:
            handles.append(model.synthesizer.decoder.feature_projection.register_forward_hook(
                lambda module, args, value: shape_checks.update(point_features=list(value.shape))))
            handles.append(model.synthesizer.decoder.net[0].register_forward_pre_hook(
                lambda module, args: shape_checks.update(mlp_input=list(args[0].shape))))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            planes = model.forward_planes(inputs)
            prediction = model.synthesizer.forward_points(planes, xyz[None].float() / 255 * 2 - 1)['sigma'].reshape(-1)
            loss = F.mse_loss(prediction, ct[tuple(xyz.T)])
        for handle in handles:
            handle.remove()
        loss.backward()
        if step == 1:
            assert shape_checks == dict(point_features=[10000, 368], mlp_input=[10000, 368])
            assert model.synthesizer.decoder.feature_projection.weight.grad.norm() > 0
            assert model.upsampler.weight.grad.norm() > 0
            assert model.synthesizer.decoder.net[0].weight.grad.norm() > 0
        torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step in (100, 300, 600, 1000):
            measure(step)
        if step % 50 == 0:
            (out / 'progress.json').write_text(json.dumps(dict(step=step, target=1000), indent=2))
    assert point_hash.hexdigest() == prior['point_samples_sha256']
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    point_generator_state=rng.get_state(), step=1000, config=OmegaConf.to_container(cfg)),
               out / 'step1000.pth')
    summary = dict(case=case, initialization='random; no model or optimizer checkpoint loaded',
                   model_seed=42, point_seed=prior['point_seed'], steps=1000,
                   point_samples_sha256=point_hash.hexdigest(), points_per_step=10000,
                   views=prior['views'], triplane_channels=128, sampled_feature_dim=384,
                   point_feature_dim=368, feature_projection='trainable Linear(384,368)',
                   mlp_widths=[368,128,128,128,1], all_parameters_trainable=True,
                   optimizer=dict(name='AdamW', lr=5e-5, betas=[.9,.95], matrix_weight_decay=.02, gradient_clip=.5),
                   evaluation='same E13 even native voxel 128^3 grid; [0,1] clipping',
                   shapes=shape_checks, metrics=rows, baseline_e13=prior['metrics'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.plot([r['step'] for r in prior['metrics']], [r['psnr'] for r in prior['metrics']], marker='o', label='E13 / 96 dimensions')
    ax.plot([r['step'] for r in rows], [r['psnr'] for r in rows], marker='s', label='128 channels per plane / projected to 368')
    ax.set(xlabel='Optimizer steps from random initialization', ylabel='Held-out volume PSNR (dB)')
    ax.legend()
    fig.savefig(out / 'curves.png', dpi=180)
    plt.close(fig)
    volumes = [('Reference',reference),('E13 / 96D',np.load(args.e13_root/'xlrm/step1000.npy')),
               ('Wide planes / 368D',np.load(out/'step1000.npy'))]
    fig, axes = plt.subplots(3,3,figsize=(8,8),layout='constrained')
    for i,(name,volume) in enumerate(volumes):
        for axis in range(3):
            axes[i,axis].imshow(np.take(volume,64,axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=.6)
            axes[i,axis].set_title(name+' / '+'XYZ'[axis])
            axes[i,axis].axis('off')
    fig.savefig(out/'comparison.png',dpi=160)
    plt.close(fig)
    (out/'phase.txt').write_text('complete\n')
    print(json.dumps(dict(status='complete',final_psnr=rows[-1]['psnr'])),flush=True)


if __name__ == '__main__':
    main()
