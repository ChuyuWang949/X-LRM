"""Continue saved E15 A model weights for 1000 updates with a fresh AdamW."""
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
    parser.add_argument('--e15-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        raise FileExistsError('Completed output already exists')
    out = args.output
    torch.set_num_threads(4)
    PartialState()
    config = OmegaConf.load(args.e15_root / 'source_config.yaml')
    config.dataset.lndb_root = str(args.dataset_root)
    old = json.loads((args.e15_root / 'summary.json').read_text())
    saved = torch.load(args.e15_root / 'A/step800.pth', map_location='cpu', weights_only=True)['model']
    model = ModelLRM(**config.model, image_res=config.dataset.input_image_res).cuda()
    model.load_state_dict(saved, strict=True)
    model.requires_grad_(True)
    model.synthesizer.decoder.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in parameters if p.ndim >= 2], weight_decay=.02),
        dict(params=[p for p in parameters if p.ndim < 2], weight_decay=0.)],
        lr=5e-5, betas=(.9, .95))
    dataset = LNDbLRMDataset(config, 'train')
    case = old['case']
    record = dataset._read_projections(dataset.names.index(case))
    inputs = dataset._encode_projections(record, np.asarray(old['source_frame_indices']))[0][None].cuda()
    ct = (torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(
        str(args.dataset_root / 'processed' / f'{case}.nii.gz')))).float() / 255).cuda()
    reference = ct[::2, ::2, ::2].contiguous().cpu().numpy()
    np.testing.assert_array_equal(reference, np.load(args.e15_root / 'heldout_ct.npy', allow_pickle=False))
    axis = torch.arange(0, 256, 2, device='cuda')
    queries = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(1, -1, 3).float() / 255 * 2 - 1
    replay_batches = torch.from_numpy(np.load(args.e15_root / 'point_batches.npy', allow_pickle=False)).cuda()
    generator = torch.Generator(device='cuda').manual_seed(old['point_seed'])
    batches = []
    for total_step in range(1, 2801):
        xyz = torch.randint(256, (12000, 3), device='cuda', generator=generator)
        xyz = xyz[(xyz % 2 == 1).any(dim=1)][:10000]
        assert xyz.shape == (10000, 3)
        if 1000 < total_step <= 1800:
            torch.testing.assert_close(xyz.to(torch.uint8), replay_batches[total_step - 1001], atol=0, rtol=0)
        if total_step > 1800:
            batches.append(xyz.to(torch.uint8))
    batches = torch.stack(batches)
    np.save(out / 'point_batches.npy', batches.cpu().numpy())
    rows = []
    losses = []
    def update(xyz):
        model.train()
        xyz = xyz.long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            planes = model.forward_planes(inputs)
            pred = model.synthesizer.forward_points(planes, xyz[None].float() / 255 * 2 - 1)['sigma'].reshape(-1)
            loss = F.mse_loss(pred, ct[tuple(xyz.T)])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def measure(step):
        model.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            planes = model.forward_planes(inputs)
            prediction = torch.cat([model.synthesizer.forward_points(planes, q)['sigma'].reshape(-1).float().cpu()
                                    for q in queries.split(131072, dim=1)]).numpy().reshape(reference.shape)
        if step == 0:
            np.testing.assert_array_equal(prediction, np.load(args.e15_root / 'A/step800.npy', allow_pickle=False))
        mse = float(np.mean((np.clip(prediction, 0, 1).astype(np.float64) - reference.astype(np.float64)) ** 2))
        row = dict(continuation_step=step, e13_total_step=1800 + step,
                   psnr=float(-10 * np.log10(mse)), mse=mse,
                   train_mse=float(np.mean(losses[-50:])) if losses else None,
                   seconds=time.monotonic() - stamp)
        rows.append(row)
        np.save(out / f'step{step}.npy', prediction)
        with (out / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    stamp = time.monotonic()
    measure(0)
    for step in range(1, 1001):
        losses.append(update(batches[step - 1]))
        if step in (100, 300, 600, 800, 1000):
            measure(step)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                            point_generator_state=generator.get_state(), next_point_batch_index=step,
                            e13_total_step=1800 + step, decoder_frozen=True), out / f'step{step}.pth')
        if step % 50 == 0:
            (out / 'progress.json').write_text(json.dumps(dict(phase='continuation', step=step, target=1000), indent=2))
    for key, value in model.synthesizer.decoder.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved['synthesizer.decoder.' + key], atol=0, rtol=0)
    summary = dict(case=case, start='E15 A/step800.pth; E13 total step 1800',
                   continuation_steps=1000, final_e13_total_step=2800,
                   decoder='same E13 decoder, frozen', points_per_step=10000,
                   point_batches_sha256=hashlib.sha256(batches.cpu().numpy().tobytes()).hexdigest(),
                   point_stream='same CUDA generator advanced through total steps 1 to 1800, then continued',
                   optimizer=dict(name='AdamW', lr=5e-5, betas=[.9,.95], matrix_weight_decay=.02,
                                  gradient_clip=.5, state='reset; E15 checkpoint contains model weights only'),
                   dropout='same train-mode dropout as E15 A',
                   optimizer_state_restored=False,
                   evaluation='same E13 128^3 held-out even native voxels, [0,1] clipping', metrics=rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    history = old['metrics']['A']
    x = [r['e13_total_step'] for r in history] + [r['e13_total_step'] for r in rows[1:]]
    y = [r['psnr'] for r in history] + [r['psnr'] for r in rows[1:]]
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.plot(x, y, marker='o')
    ax.axvline(1800, color='gray', linestyle='--', linewidth=1)
    ax.set(xlabel='Total optimizer updates since random initialization', ylabel='Held-out volume PSNR (dB)')
    fig.savefig(out / 'curves.png', dpi=180)
    plt.close(fig)
    (out / 'phase.txt').write_text('complete\n')
    print(json.dumps(dict(status='complete', final_psnr=rows[-1]['psnr'])), flush=True)


if __name__ == '__main__':
    main()
