"""Native XLRM training loop with the shared LNDb patient splits."""
import json
from pathlib import Path
import sys

from safetensors.torch import load_model, save_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from x_ggrm.datasets.lndb_lrm import LNDbLRMDataset
from x_ggrm.models.rendering.utils.ray_sampler import RaySampler
from .lrm import CbctLrmTrainer


class LNDbLRMTrainer(CbctLrmTrainer):
    def _build_model(self, cfg):
        self.ct_points_per_case = int(cfg.train.get('ct_points_per_case', 0))
        self.ct_volume_only = bool(cfg.train.get('ct_volume_only', False))
        if self.ct_points_per_case < 0:
            raise ValueError('CT point count must be nonnegative')
        if self.ct_points_per_case and self.ct_volume_only:
            raise ValueError('Choose CT point or full CT volume supervision')
        if cfg.model.get('reprojection_channels') and not (self.ct_points_per_case or self.ct_volume_only):
            raise ValueError('Reprojection branch requires CT-only supervision')
        if (self.ct_points_per_case or self.ct_volume_only) and (
            cfg.train.loss.pixel_weight != 0 or cfg.train.loss.tv_weight != 0
        ):
            raise ValueError('CT-only supervision requires pixel and TV losses to be disabled')
        model = super()._build_model(cfg)
        self.teacher = None
        distillation = cfg.get('distillation')
        self.distill_weight = float(distillation.weight) if distillation else 0.
        if self.distill_weight < 0:
            raise ValueError('Distillation weight must be nonnegative')
        if self.distill_weight:
            if not distillation.teacher_checkpoint:
                raise ValueError('Distillation requires a trained dense-view teacher checkpoint')
            # Start both models in the same channel basis and keep the decoder shared in value.
            cfg.saver.load_model = str(distillation.teacher_checkpoint)
        if self.distill_weight or cfg.train.get('freeze_point_decoder', False):
            model.synthesizer.decoder.requires_grad_(False)
        return model

    def prepare_everything(self, is_dist_validation=True):
        super().prepare_everything(is_dist_validation)
        if self.distill_weight:
            # Preserve the student RNG stream for the equal-budget control.
            with torch.random.fork_rng(devices=[]):
                self.teacher = type(self.accelerator.unwrap_model(self.model))(
                    **self.cfg.model, image_res=self.cfg.dataset.input_image_res
                )
            load_model(self.teacher, self.cfg.distillation.teacher_checkpoint, strict=True)
            self.teacher.to(self.device).requires_grad_(False).eval()

    def train(self):
        best_root = self.cfg.saver.get('best_volume_root')
        if best_root and self.global_step == 0 and not (Path(best_root) / 'selection.json').is_file():
            self.evaluate()
        super().train()

    def evaluate(self, epoch=None):
        metrics = super().evaluate(epoch)
        root = self.cfg.saver.get('best_volume_root')
        if root and self.accelerator.is_main_process:
            self._keep_best_volume(Path(root), metrics)
        return metrics

    def _keep_best_volume(self, root, metrics):
        score = metrics['loss_vol_mse']
        selection = root / 'selection.json'
        if selection.is_file() and score >= json.loads(selection.read_text())['validation_volume_mse']:
            return
        folder = root / f'{self.global_step:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / 'model.safetensors'
        temporary = checkpoint.with_suffix('.tmp')
        save_model(self.accelerator.unwrap_model(self.model), str(temporary))
        temporary.replace(checkpoint)
        record = dict(step=self.global_step, validation_volume_mse=score, checkpoint=str(checkpoint.resolve()))
        pending = selection.with_suffix('.tmp')
        pending.write_text(json.dumps(record, indent=2) + '\n')
        pending.replace(selection)
        print(f'[BEST VOLUME] step={self.global_step} validation_volume_mse={score:.8f}', flush=True)

    def _build_dataloader(self, cfg):
        def make(split, batch, workers):
            return DataLoader(LNDbLRMDataset(cfg, split), batch_size=batch,
                              shuffle=split == 'train', drop_last=split == 'train',
                              num_workers=workers, pin_memory=True,
                              persistent_workers=split == 'train' and workers > 0)
        return (make('train', cfg.train.batch_size, cfg.dataset.num_train_workers),
                make('eval', cfg.val.batch_size, cfg.dataset.num_val_workers))

    def forward_loss_local_step(self, data):
        if self.ct_points_per_case or self.ct_volume_only:
            outputs = self.model(image=data['input_views'], ct_query_points=data.get('ct_query_points'),
                                 input_cameras=data.get('input_cameras'))
            if self.model.training and self.ct_points_per_case:
                volume = F.mse_loss(outputs['ct_point_pred'], data['ct_point_targets'])
            else:
                volume = F.mse_loss(outputs['vol_pred'], data['vol_gt'])
            loss, pixel, tv = self.cfg.train.loss.vol_mse_weight * volume, None, None
        else:
            outputs, loss, pixel, tv, volume = super().forward_loss_local_step(data)
        if self.teacher is not None and self.model.training:
            with torch.no_grad(), self.accelerator.autocast():
                teacher_planes = self.teacher.forward_planes(data['teacher_input_views'])
            planes = outputs['planes'].float()
            if planes.shape != teacher_planes.shape:
                raise ValueError('Teacher and student must share the same triplane grid')
            feature_loss = F.l1_loss(planes, teacher_planes.float())
            outputs['loss_triplane_distillation'] = feature_loss
            loss = loss + self.distill_weight * feature_loss
            if self.accelerator.sync_gradients:
                self.accelerator.log({'train/loss_triplane_distillation': feature_loss.detach().item()},
                                     step=self.global_step + 1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {self.global_step}')
        return outputs, loss, pixel, tv, volume


def smoke(trainer):
    batch = next(iter(trainer.train_loader))
    training = trainer.model
    training.train()
    if not (trainer.ct_points_per_case or trainer.ct_volume_only):
        render = batch['render_cameras'].float()[0]
        origins, rays = RaySampler()(render[:, :16].reshape(-1,4,4),
                                     render[:, 16:].reshape(-1,3,3),
                                     batch['render_full_resolutions'][0], batch['render_anchors'][0],
                                     trainer.cfg.dataset.render_image.region)
        # The central ray of each calibrated camera must intersect the CT isocenter.
        pose = render[:, :16].reshape(-1,4,4)
        central = -pose[:, :3, 2]
        torch.testing.assert_close(origins[:, 0], pose[:, :3, 3])
        torch.testing.assert_close(central, -torch.nn.functional.normalize(origins[:, 0], dim=-1))
        assert torch.isfinite(rays).all()
    tracked = trainer.accelerator.unwrap_model(training).upsampler.weight
    before = tracked.detach().clone()
    trainer.optimizer.zero_grad()
    trainer.optimizer.param_groups[0]['lr'] = trainer.cfg.train.optim.lr
    trainer.optimizer.param_groups[1]['lr'] = trainer.cfg.train.optim.lr
    output, loss, pixel, tv, volume = trainer.forward_loss_local_step(batch)
    if trainer.ct_points_per_case:
        import SimpleITK as sitk
        dataset = trainer.train_loader.dataset
        reference = sitk.GetArrayFromImage(sitk.ReadImage(str(dataset.root / 'processed' / (batch['id'][0] + '.nii.gz'))))
        coords = ((batch['ct_query_points'][0] + 1) * 127.5).round().cpu().long().numpy()
        expected = torch.from_numpy(reference[tuple(coords.T)])
        sampled = (batch['ct_point_targets'][0] * 255).round().to(torch.uint8).cpu()
        torch.testing.assert_close(expected, sampled, atol=0, rtol=0)
        if output['ct_point_pred'].shape != batch['ct_point_targets'].shape or 'images_density' in output:
            raise ValueError('Expected point densities only')
    elif trainer.ct_volume_only:
        if output['vol_pred'].shape != batch['vol_gt'].shape or 'images_density' in output:
            raise ValueError('Expected an aligned CT volume without projection rendering')
    elif output['vol_pred'].shape != batch['vol_gt'].shape or output['images_density'].shape != batch['render_images'].shape:
        raise ValueError('Prediction and target shapes differ')
    distill_stats = {}
    if trainer.teacher is not None:
        sparse = batch['input_view_indices'][0].contiguous()
        dense = batch['teacher_view_indices'][0].contiguous()
        positions = torch.searchsorted(dense, sparse)
        torch.testing.assert_close(dense[positions], sparse)
        torch.testing.assert_close(batch['teacher_input_views'][0, positions], batch['input_views'][0],
                                   atol=0, rtol=0)
        feature_loss = output['loss_triplane_distillation']
        grad, = torch.autograd.grad(feature_loss, tracked, retain_graph=True)
        grad_norm = grad.float().norm().item()
        if grad_norm == 0 or not torch.isfinite(grad).all():
            raise RuntimeError('Triplane loss did not produce a finite student gradient')
        distill_stats = dict(sparse_ids=sparse.tolist(), dense_ids=dense.tolist(),
                             triplane_shape=list(output['planes'].shape),
                             distillation_l1=feature_loss.item(), distillation_grad_norm=grad_norm)
    trainer.accelerator.backward(loss)
    if trainer.teacher is not None:
        if any(p.grad is not None for p in trainer.teacher.parameters()):
            raise RuntimeError('Teacher received gradients')
        decoder = trainer.accelerator.unwrap_model(training).synthesizer.decoder
        if any(p.grad is not None for p in decoder.parameters()):
            raise RuntimeError('Frozen student decoder received gradients')
        distill_stats.update(teacher_has_grad=False, student_decoder_has_grad=False)
    norm = torch.nn.utils.clip_grad_norm_(training.parameters(), 0.5, error_if_nonfinite=True)
    trainer.optimizer.step()
    if torch.equal(tracked, before):
        raise RuntimeError('No optimizer update')
    row = dict(status='passed', case=batch['id'], loss=float(loss), projection_mse=float(pixel) if pixel is not None else None,
               volume_mse=float(volume) if volume is not None else None, grad_norm=float(norm), prediction_shape=list(output.get('vol_pred', output.get('ct_point_pred')).shape),
               peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, **distill_stats)
    if 'vol_pred' in output:
        row['volume_shape'] = list(output['vol_pred'].shape)
    if trainer.teacher is not None or trainer.ct_points_per_case or trainer.ct_volume_only:
        training.eval()
        validation = next(iter(trainer.val_loader))
        if 'teacher_input_views' in validation:
            raise RuntimeError('Validation data contains teacher input')
        with torch.no_grad():
            _, val_loss, _, _, _ = trainer.forward_loss_local_step(validation)
        key = 'sparse_only_validation_loss' if trainer.teacher is not None else 'validation_volume_loss'
        row[key] = val_loss.item()
    Path('smoke.json').write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    if is_smoke:
        sys.argv.remove('--smoke')
    torch.set_num_threads(4)
    with LNDbLRMTrainer() as trainer:
        if is_smoke:
            smoke(trainer)
        else:
            trainer.run()
