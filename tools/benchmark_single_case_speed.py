"""Synchronized XLRM training benchmarks at unchanged case, views and CT points."""
import argparse
import json
from pathlib import Path
import time

from accelerate import PartialState
import numpy as np
from omegaconf import OmegaConf
import SimpleITK as sitk
import torch
from torch.nn import functional as F

from x_ggrm.datasets.lndb_lrm import LNDbLRMDataset
from x_ggrm.models import ModelLRM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    PartialState()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    summary = json.loads((args.checkpoint.parent/'summary.json').read_text())
    cfg = OmegaConf.create(saved['config'])
    cfg.dataset.lndb_root = str(args.dataset_root)
    ds = LNDbLRMDataset(cfg, 'train')
    record = ds._read_projections(ds.names.index(summary['case']))
    inputs, _, pose = ds._encode_projections(record, np.asarray(summary['views']))
    inputs, cameras = inputs[None].cuda(), ds.pack_cameras(pose)[None].cuda()
    projections = inputs[..., :1].permute(0, 1, 4, 2, 3)
    ct = torch.as_tensor(sitk.GetArrayFromImage(sitk.ReadImage(
        str(args.dataset_root/'processed'/f'{summary["case"]}.nii.gz'))).astype(np.float32), device='cuda')/255
    model = ModelLRM(**cfg.model, image_res=cfg.dataset.input_image_res).cuda().train()
    parameters = list(model.parameters())
    rng = torch.Generator(device='cuda')
    labels = ['sample', 'planes', 'local_cnn', 'query_loss', 'backward', 'clip_adamw']

    def reset(checkpointing, fused, autotune=False):
        model.load_state_dict(saved['model'], strict=True)
        model.activation_checkpointing = checkpointing
        model.train()
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        rng.set_state(saved['point_generator_state'])
        torch.backends.cudnn.benchmark = autotune
        optimizer = torch.optim.AdamW([
            dict(params=[p for p in parameters if p.ndim>=2], weight_decay=.02),
            dict(params=[p for p in parameters if p.ndim<2], weight_decay=0.)],
            lr=5e-5, betas=(.9,.95), fused=fused)
        # Non-capturable AdamW otherwise aliases CPU step tensors across resets.
        optimizer_state = {key:dict(value, step=value['step'].clone())
                           for key,value in saved['optimizer']['state'].items()}
        optimizer.load_state_dict(dict(state=optimizer_state, param_groups=[
            dict(group, fused=fused, foreach=False if fused else None)
            for group in saved['optimizer']['param_groups']]))
        return optimizer

    def step(optimizer, timing=False, inspect=False):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(7)] if timing else None
        def mark(i):
            if events:
                events[i].record()
        mark(0)
        xyz = torch.randint(256, (12000,3), device='cuda', generator=rng)
        xyz = xyz[(xyz%2==1).any(dim=1)][:10000]
        points = xyz[None].float()/255*2-1
        target = ct[tuple(xyz.T)][None]
        optimizer.zero_grad(set_to_none=True)
        mark(1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.profiler.record_function('encode_planes'):
                planes = model.forward_planes(inputs)
            mark(2)
            with torch.profiler.record_function('encode_local'):
                local = model.reprojection_encoder(projections)
            mark(3)
            context = dict(planes=planes, reprojection_features=local, input_cameras=cameras)
            with torch.profiler.record_function('query_points'):
                pred = model.query_points(context, points)['sigma']
                loss = F.mse_loss(pred, target)
        mark(4)
        with torch.profiler.record_function('backward'):
            loss.backward()
        mark(5)
        if inspect:
            inspected = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(parameters,.5,error_if_nonfinite=True)
        optimizer.step()
        mark(6)
        scalar = float(loss.detach())  # Preserve the original per-step CPU synchronization.
        if inspect:
            return scalar, pred.detach().float().cpu(), inspected, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        return events

    # Check the proposed no-checkpoint + fused path against one baseline update.
    opt = reset(True, False)
    reference = step(opt, inspect=True)
    del opt
    opt = reset(True, False)
    repeat = step(opt, inspect=True)
    del opt
    opt = reset(False, True)
    candidate = step(opt, inspect=True)
    def difference(other):
        error = {k:(reference[2][k]-other[2][k]).abs().max().item() for k in reference[2]}
        squared = sum(float((reference[2][k]-other[2][k]).square().sum()) for k in reference[2])
        norm = sum(float(v.square().sum()) for v in reference[2].values())
        return dict(prediction_max_abs_error=(reference[1]-other[1]).abs().max().item(),
                    loss_abs_error=abs(reference[0]-other[0]),gradient_max_abs_error=max(error.values()),
                    gradient_worst_parameter=max(error,key=error.get),gradient_relative_l2=(squared/norm)**.5,
                    updated_state_max_abs_error=max((reference[3][k]-other[3][k]).abs().max().item()
                        for k in reference[3] if reference[3][k].is_floating_point()))
    control, proposed = difference(repeat), difference(candidate)
    correctness = dict(baseline_repeat=control, no_checkpoint_fused=proposed)
    print(json.dumps(correctness),flush=True)
    assert proposed['prediction_max_abs_error']==0
    assert proposed['gradient_relative_l2'] <= max(1e-3,2*control['gradient_relative_l2'])
    assert proposed['updated_state_max_abs_error'] <= max(2e-6,2*control['updated_state_max_abs_error'])
    del reference, repeat, candidate, opt
    measurements = []
    for name,checkpointing,fused,autotune in [
        ('baseline',True,False,False), ('no_checkpoint',False,False,False),
        ('fused_only',True,True,False), ('no_checkpoint_fused',False,True,False),
        ('no_checkpoint_fused_autotune',False,True,True)]:
        opt = reset(checkpointing,fused,autotune)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        warm_start = time.perf_counter()
        for _ in range(args.warmup):
            step(opt)
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter()-warm_start
        start = time.perf_counter()
        all_events = [step(opt, timing=True) for _ in range(args.steps)]
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        stages = {label:float(np.mean([e[i].elapsed_time(e[i+1]) for e in all_events])) for i,label in enumerate(labels)}
        row=dict(name=name,checkpointing=checkpointing,fused_adamw=fused,cudnn_benchmark=autotune,
                 wall_ms_per_step=elapsed/args.steps*1000,stage_cuda_ms=stages,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,warmup_seconds=warm_seconds)
        measurements.append(row)
        print(json.dumps(row),flush=True)
        del opt
    opt = reset(True,False)
    for _ in range(3):
        step(opt)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step(opt)
    (args.output/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,views=len(summary['views']),
                points_per_step=10000,warmup=args.warmup,timed_steps=args.steps,
                initialization='checkpoint model, AdamW and RNG restored for each timing arm',
                correctness=correctness,measurements=measurements)
    (args.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(correctness),flush=True)


if __name__=='__main__':
    main()
