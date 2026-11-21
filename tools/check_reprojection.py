"""Validate reprojection geometry, feature queries, gradients and legacy checkpoints."""
import argparse
import json
from pathlib import Path

from accelerate import PartialState
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from x_ggrm.datasets.lndb_lrm import LNDbLRMDataset
from x_ggrm.models import ModelLRM
from x_ggrm.models.reprojection import project_points, sample_reprojection_features
from x_ggrm.models.rendering.utils.ray_sampler import RaySampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--legacy-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    PartialState()
    torch.manual_seed(42)
    cfg = OmegaConf.load(args.config)
    cfg.dataset.lndb_root = str(args.dataset_root)
    cfg.model.volume_grid_size = 8
    dataset = LNDbLRMDataset(cfg, 'train')
    record = dataset._read_projections(dataset.names.index('LNDb-0045'))
    ids = np.linspace(0, len(record['angles']), 24, endpoint=False).astype(int)
    images, _, poses = dataset._encode_projections(record, ids)
    images = images[None].cuda()
    cameras = dataset.pack_cameras(poses)[None].cuda()
    points = torch.rand(1, 10000, 3, device='cuda') * 2 - 1
    result = dict(case='LNDb-0045', views=ids.tolist(), camera='Blender world_from_camera + normalized OpenCV K',
                  world_units='XLRM [-1,1] cube; source translation uses the same units',
                  sampling='column,row; align_corners=False; max over valid views')

    actual, valid = project_points(points, cameras)
    geometry = dataset.geometry
    x, y, z = (points[0] * (geometry['sVoxel'][0] / 2)).unbind(-1)
    angle = torch.as_tensor(record['angles'][ids], device='cuda', dtype=torch.float32)[:, None]
    xr = angle.cos() * x + angle.sin() * y
    yr = -angle.sin() * x + angle.cos() * y
    scale = 2 * geometry['DSD'] / (geometry['DSO'] - xr)
    analytical = torch.stack((yr * scale / geometry['sDetector'][1],
                              -z * scale / geometry['sDetector'][0]), dim=-1)[None]
    torch.testing.assert_close(actual, analytical, atol=1e-6, rtol=0)
    result['analytical_projection_max_abs_error'] = (actual - analytical).abs().max().item()
    centre, _ = project_points(torch.zeros(1, 1, 3, device='cuda'), cameras)
    torch.testing.assert_close(centre, torch.zeros_like(centre), atol=1e-6, rtol=0)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        amp_grid, _ = project_points(points, cameras)
    torch.testing.assert_close(amp_grid, actual, atol=0, rtol=0)
    result['autocast_geometry_max_abs_error'] = (amp_grid - actual).abs().max().item()

    # Independently invert the existing source-ray sampler at noncentral pixels.
    selected = cameras[0, [0, 12, 23]]
    anchors = torch.tensor([[21, 81], [104, 14], [168, 162]], device='cuda')
    origins, rays = RaySampler()(selected[:, :16].reshape(-1, 4, 4),
                                selected[:, 16:].reshape(-1, 3, 3),
                                torch.full((3, 1), 224, device='cuda'), anchors, 4)
    ray_points = origins + 2 * rays
    recovered, _ = project_points(ray_points, selected[:, None])
    yy, xx = torch.meshgrid(torch.arange(4, device='cuda'), torch.arange(4, device='cuda'), indexing='ij')
    expected = (torch.stack((xx, yy), dim=-1).reshape(1, -1, 2) + anchors[:, None].flip(-1) + .5) / 224 * 2 - 1
    torch.testing.assert_close(recovered[:, 0], expected, atol=1e-6, rtol=0)
    result['ray_roundtrip_max_abs_error'] = (recovered[:, 0] - expected).abs().max().item()
    axis = (torch.arange(224, device='cuda') + .5) / 224 * 2 - 1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    ramp = torch.stack((xx, yy))[None, None].expand(3, 1, -1, -1, -1)
    sampled = sample_reprojection_features([ramp], ray_points, selected[:, None])
    torch.testing.assert_close(sampled, expected, atol=1e-6, rtol=0)
    result['pixel_ramp_max_abs_error'] = (sampled - expected).abs().max().item()
    # Points behind every camera must yield finite zeros, including for negative features.
    behind = selected[:, None, :16].reshape(3, 1, 4, 4)[..., :3, 3] + selected[:, None, :16].reshape(3, 1, 4, 4)[..., :3, 2]
    invisible = sample_reprojection_features([-torch.ones_like(ramp)], behind, selected[:, None])
    assert torch.equal(invisible, torch.zeros_like(invisible))
    result['all_invisible_points_zero'] = True

    # Actual dataset item must carry exactly the cameras of its selected input frames.
    item = dataset[dataset.names.index('LNDb-0045')]
    assert item['input_cameras'].shape == (24, 25)
    origin = item['input_cameras'][:, :16].reshape(24, 4, 4)[:, :3, 3]
    torch.testing.assert_close(item['input_views'][:, 0, 0, 1:4], origin)
    result['dataset_camera_image_order'] = 'passed'

    # Original parameter names/shapes and predictions remain usable with the branch disabled.
    saved = torch.load(args.legacy_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    legacy_config = dict(saved['config']['model'])
    legacy_config['reprojection_channels'] = None
    legacy = ModelLRM(**legacy_config, image_res=224).cuda().eval()
    legacy.load_state_dict(saved['model'], strict=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        query = points[:, :1000]
        old = legacy.synthesizer.forward_points(legacy.forward_planes(images), query)['sigma']
        new = legacy(images, ct_query_points=query)['ct_point_pred']
    torch.testing.assert_close(new, old, atol=0, rtol=0)
    result['legacy_strict_load'] = 'passed'
    result['legacy_forward_max_abs_error'] = (new - old).abs().max().item()
    del saved, legacy
    torch.cuda.empty_cache()

    model = ModelLRM(**cfg.model, image_res=224).cuda()
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(images, input_cameras=cameras, ct_query_points=points)
        # Real CT supervision at the nearest native grid points.
        import SimpleITK as sitk
        ct = torch.as_tensor(sitk.GetArrayFromImage(sitk.ReadImage(
            str(args.dataset_root / 'processed/LNDb-0045.nii.gz'))).astype(np.float32), device='cuda') / 255
        xyz = ((points[0] + 1) * 127.5).round().long()
        target = ct[tuple(xyz.T)][None]
        loss = F.mse_loss(output['ct_point_pred'], target)
    loss.backward()
    grads = dict(local_encoder=model.reprojection_encoder.stages[0][0].weight.grad.norm().item(),
                 triplane_upsampler=model.upsampler.weight.grad.norm().item(),
                 decoder=model.synthesizer.decoder.net[0].weight.grad.norm().item(),
                 local_decoder_columns=model.synthesizer.decoder.net[0].weight.grad[:, 3*cfg.model.triplane_dim:].norm().item())
    assert all(np.isfinite(value) and value > 0 for value in grads.values())
    result.update(gradient_norms=grads, ct_mse=loss.item(), decoder_input_dim=model.synthesizer.decoder.net[0].in_features)
    assert result['decoder_input_dim'] == 3*cfg.model.triplane_dim + sum(cfg.model.reprojection_channels)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        context = model.encode(images, cameras)
        result['local_map_shapes'] = [list(v.shape) for v in context['reprojection_features']]
        local = sample_reprojection_features(context['reprojection_features'], points[:, :257], cameras)
        permutation = torch.randperm(24, device='cuda')
        shuffled = sample_reprojection_features([f[:, permutation] for f in context['reprojection_features']],
                                               points[:, :257], cameras[:, permutation])
        torch.testing.assert_close(local, shuffled, atol=0, rtol=0)
        blocked = sample_reprojection_features(context['reprojection_features'], points[:, :257], cameras,
                                              torch.zeros(1, 24, device='cuda', dtype=torch.bool))
        assert torch.equal(blocked, torch.zeros_like(blocked))
        part = model.query_points(context, points[:, :1000], chunk_size=257)['sigma']
        whole = model.query_points(context, points[:, :1000])['sigma']
        # BF16 GEMM kernels can round by one output ULP at different batch sizes.
        result['bf16_chunk_max_abs_error'] = (part - whole).abs().max().item()
        precise_context = dict(context, planes=context['planes'].float(),
                               reprojection_features=[f.float() for f in context['reprojection_features']])
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        with torch.autocast('cuda', enabled=False):
            part = model.query_points(precise_context, points[:, :1000], chunk_size=257)['sigma']
            whole = model.query_points(precise_context, points[:, :1000])['sigma']
        torch.set_float32_matmul_precision(precision)
        torch.testing.assert_close(part, whole, atol=2e-6, rtol=0)
        axis = torch.linspace(-1, 1, 8, device='cuda')
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)[None]
        direct = model.query_points(context, grid.reshape(1, -1, 3))['sigma'].reshape(1, 8, 8, 8)
        volume = model(images, input_cameras=cameras)['vol_pred']
        torch.testing.assert_close(volume, direct, atol=0, rtol=0)
        # An intensity change in one source scan must change its local features.
        changed = model.reprojection_encoder(torch.zeros_like(images[..., :1].permute(0, 1, 4, 2, 3)))
        changed_local = sample_reprojection_features(changed, points[:, :257], cameras)
        assert (changed_local - local).abs().max() > 0
    result.update(view_permutation_invariant=True, view_mask_zero=True,
                  chunk_max_abs_error=(part - whole).abs().max().item(),
                  grid_point_max_abs_error=(volume - direct).abs().max().item(),
                  input_change_local_feature_max_abs_delta=(changed_local-local).abs().max().item(),
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, status='passed')
    try:
        model.encode(images)
    except ValueError:
        result['missing_geometry_rejected'] = True
    else:
        raise AssertionError('Missing geometry was silently accepted')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
