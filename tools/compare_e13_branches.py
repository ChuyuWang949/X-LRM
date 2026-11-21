"""Continue E13 with a shared frozen decoder and matched CT point batches."""
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
from torch import nn
from torch.nn import functional as F

from x_ggrm.datasets.lndb_lrm import LNDbLRMDataset
from x_ggrm.models import ModelLRM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e13-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    PartialState()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    cfg = OmegaConf.load(args.config)
    cfg.dataset.lndb_root = str(args.dataset_root)
    previous = json.loads((args.e13_root / 'xlrm/summary.json').read_text())
    state = torch.load(args.e13_root / 'xlrm/step1000.pth', map_location='cpu', weights_only=True)
    model = ModelLRM(**cfg.model, image_res=cfg.dataset.input_image_res).cuda()
    model.load_state_dict(state, strict=True)
    dataset = LNDbLRMDataset(cfg, 'train')
    case = previous['case']
    record = dataset._read_projections(dataset.names.index(case))
    frames = np.asarray(previous['views'])
    inputs = dataset._encode_projections(record, frames)[0][None].cuda()
    raw_ct = sitk.GetArrayFromImage(sitk.ReadImage(str(args.dataset_root / 'processed' / f'{case}.nii.gz')))
    ct = (torch.from_numpy(raw_ct).float() / 255).cuda()
    reference = ct[::2, ::2, ::2].contiguous().cpu().numpy()
    np.testing.assert_array_equal(reference, np.load(args.e13_root / 'heldout_ct.npy', allow_pickle=False))
    np.save(out / 'heldout_ct.npy', reference)
    axis = torch.arange(0, 256, 2, device='cuda')
    queries = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(1, -1, 3).float() / 255 * 2 - 1

    generator = torch.Generator(device='cuda').manual_seed(previous['point_seed'])
    original_hash = hashlib.sha256()
    point_batches = []
    for step in range(1, 1801):
        xyz = torch.randint(256, (12000, 3), device='cuda', generator=generator)
        xyz = xyz[(xyz % 2 == 1).any(dim=1)][:10000]
        assert xyz.shape == (10000, 3)
        if step in (1, 100, 300, 600, 1000):
            original_hash.update(xyz.cpu().numpy().tobytes())
        if step > 1000:
            point_batches.append(xyz.to(torch.uint8))
    assert original_hash.hexdigest() == previous['point_samples_sha256']
    point_batches = torch.stack(point_batches)
    np.save(out / 'point_batches.npy', point_batches.cpu().numpy())
    batch_hash = hashlib.sha256(point_batches.cpu().numpy().tobytes()).hexdigest()

    captured = []
    handle = model.transformer.register_forward_hook(lambda module, args, value: captured.append(value.detach().float().clone()))
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        initial_planes = model.forward_planes(inputs).detach().float().clone()
    handle.remove()
    initial_tokens, = captured
    assert initial_tokens.shape == (1, 3 * 32 * 32, 512)
    assert initial_planes.shape == (1, 3, 32, 64, 64)
    torch.save(dict(tokens=initial_tokens.cpu(), planes=initial_planes.cpu()), out / 'initial_features.pth')
    # A non-overlapping 2x2 transposed convolution is a 512-to-128 map per token.
    upsample = model.upsampler
    assert upsample.kernel_size == upsample.stride == (2, 2)
    matrix = upsample.weight.detach().cpu().double().reshape(512, 128).T
    singular = torch.linalg.svdvals(matrix)
    upsample_diagnostic = dict(
        matrix_shape=list(matrix.shape), rank=int(torch.linalg.matrix_rank(matrix)),
        minimum_singular_value=float(singular[-1]), maximum_singular_value=float(singular[0]),
        condition_number=float(singular[0] / singular[-1]),
        interpretation='Full row rank permits any 2x2x32 output block with unconstrained 512-channel tokens.')
    np.save(out / 'upsampler_singular_values.npy', singular.numpy())

    rows = []
    baseline_prediction = None
    qa = {}
    completed = []
    checks = (0, 100, 300, 800)

    def run(arm):
        nonlocal baseline_prediction
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model.load_state_dict(state, strict=True)
        model.requires_grad_(arm == 'A')
        model.synthesizer.decoder.requires_grad_(False)
        model.eval()
        feature = None
        if arm in ('B', 'B_unfrozen'):
            feature = nn.Parameter(initial_tokens.clone())
            model.upsampler.requires_grad_(arm == 'B_unfrozen')
        elif arm == 'C':
            feature = nn.Parameter(initial_planes.clone())
        params = [p for p in model.parameters() if p.requires_grad]
        if feature is not None:
            params.append(feature)
        optimizer = torch.optim.AdamW([
            dict(params=[p for p in params if p.ndim >= 2], weight_decay=.02),
            dict(params=[p for p in params if p.ndim < 2], weight_decay=0.)],
            lr=5e-5, betas=(.9, .95))
        start = time.monotonic()
        losses = []
        history = []
        folder = out / arm
        folder.mkdir()

        def planes():
            if arm == 'A':
                return model.forward_planes(inputs)
            if arm in ('B', 'B_unfrozen'):
                return model.reshape_upsample(feature)
            return feature

        @torch.no_grad()
        def measure(step):
            nonlocal baseline_prediction
            model.eval()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = planes()
                pred = torch.cat([model.synthesizer.forward_points(features, chunk)['sigma'].reshape(-1).float().cpu()
                                  for chunk in queries.split(131072, dim=1)]).numpy().reshape(reference.shape)
            if step == 0:
                if baseline_prediction is None:
                    baseline_prediction = pred.copy()
                    np.testing.assert_allclose(pred, np.load(args.e13_root / 'xlrm/step1000.npy'), atol=2e-6, rtol=0)
                error = float(np.abs(pred - baseline_prediction).max())
                assert error <= 2e-6, (arm, error)
                qa[arm] = dict(initial_max_abs_prediction_difference=error,
                               decoder_frozen=True, trainable_parameters=sum(p.numel() for p in params))
            clipped = np.clip(pred, 0, 1)
            mse = float(np.mean((clipped.astype(np.float64) - reference.astype(np.float64)) ** 2))
            row = dict(arm=arm, additional_step=step, e13_total_step=1000 + step,
                       psnr=float(-10 * np.log10(mse)), mse=mse,
                       train_mse=float(np.mean(losses[-50:])) if losses else None,
                       seconds=time.monotonic() - start)
            rows.append(row)
            history.append(row)
            with (out / 'metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            np.save(folder / f'step{step}.npy', pred)
            print(json.dumps(row), flush=True)

        measure(0)
        for step in range(1, 801):
            model.train(arm == 'A')
            xyz = point_batches[step - 1].long()
            q = xyz[None].float() / 255 * 2 - 1
            target = ct[tuple(xyz.T)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model.synthesizer.forward_points(planes(), q)['sigma'].reshape(-1)
                loss = F.mse_loss(pred, target)
            loss.backward()
            assert all(p.grad is None for p in model.synthesizer.decoder.parameters())
            torch.nn.utils.clip_grad_norm_(params, .5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step in checks:
                measure(step)
            if step % 50 == 0:
                (out / 'progress.json').write_text(json.dumps(dict(arm=arm, step=step, target=800, completed=completed), indent=2))
        for key, value in model.synthesizer.decoder.state_dict().items():
            torch.testing.assert_close(value.cpu(), state['synthesizer.decoder.' + key], atol=0, rtol=0)
        if arm == 'B':
            for key, value in model.upsampler.state_dict().items():
                torch.testing.assert_close(value.cpu(), state['upsampler.' + key], atol=0, rtol=0)
        checkpoint = dict(arm=arm, step=800, point_batches_sha256=batch_hash,
                          decoder=model.synthesizer.decoder.state_dict())
        if arm == 'A':
            checkpoint['model'] = model.state_dict()
        else:
            checkpoint['feature'] = feature.detach().cpu()
            if arm.startswith('B'):
                checkpoint['upsampler'] = model.upsampler.state_dict()
        torch.save(checkpoint, folder / 'step800.pth')
        completed.append(arm)
        return history

    results = {arm: run(arm) for arm in ['A', 'B', 'C']}
    c_minus_b = results['C'][-1]['psnr'] - results['B'][-1]['psnr']
    if c_minus_b >= .5:
        results['B_unfrozen'] = run('B_unfrozen')
    summary = dict(
        case=case, source_checkpoint='E13 xlrm/step1000.pth', additional_steps=800,
        source_frame_indices=frames.tolist(), decoder='identical checkpoint decoder, frozen in every arm',
        train_points_per_step=10000, point_seed=previous['point_seed'],
        point_stream='E13 generator advanced through the original 1000 updates, then continued for 800 updates',
        point_batches_sha256=batch_hash, heldout='same E13 even/even/even native voxels; 128^3; [0,1] clipping',
        optimizer=dict(name='AdamW', state='reset in every arm; E13 has no saved optimizer state',
                       lr=5e-5, betas=[.9,.95], matrix_weight_decay=.02, gradient_clip=.5),
        stochastic_layers='A keeps E13 train-mode dropout; direct-feature arms use fixed deterministic downstream modules',
        precision='BF16 autocast; trainable detached features stored in FP32',
        starting_tensor_shapes=dict(tokens=list(initial_tokens.shape), planes=list(initial_planes.shape)),
        conditional_b_unfrozen_threshold_db=.5, c_minus_b_at_800_db=c_minus_b,
        qa=qa, upsampler=upsample_diagnostic, metrics=results)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    for arm, history in results.items():
        ax.plot([r['additional_step'] for r in history], [r['psnr'] for r in history], marker='o', label=arm)
    ax.set(xlabel='Additional optimizer steps after E13 step 1000', ylabel='Held-out volume PSNR (dB)')
    ax.legend()
    fig.savefig(out / 'curves.png', dpi=180)
    plt.close(fig)
    volumes = [('Reference', reference)] + [(arm, np.load(out / arm / 'step800.npy')) for arm in results]
    fig, axes = plt.subplots(len(volumes), 3, figsize=(8, len(volumes) * 2.4), layout='constrained')
    for i, (label, volume) in enumerate(volumes):
        for axis in range(3):
            axes[i, axis].imshow(np.take(volume, 64, axis=axis).T, cmap='gray', origin='lower', vmin=0, vmax=.6)
            axes[i, axis].set_title(label + ' / ' + 'XYZ'[axis])
            axes[i, axis].axis('off')
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)
    (out / 'phase.txt').write_text('complete\n')
    print(json.dumps(dict(status='complete', c_minus_b_at_800_db=c_minus_b, upsampler=upsample_diagnostic)), flush=True)


if __name__ == '__main__':
    main()
