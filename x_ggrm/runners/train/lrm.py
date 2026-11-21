# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for X-LRM for CBCT reconstruction.

import os
from abc import ABC

import math
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from accelerate.logging import get_logger

from .base_trainer import Trainer
from x_ggrm.runners import REGISTRY_RUNNERS
from x_ggrm.utils.profiler_utils import DummyProfiler

logger = get_logger(__name__)


@REGISTRY_RUNNERS.register('train.lrm')
class CbctLrmTrainer(Trainer, ABC):
    def __init__(self):
        super().__init__()
        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self.pixel_loss_fn, self.tv_loss_fn = self._build_loss_fn(self.cfg)

    def _build_model(self, cfg):
        assert cfg.experiment.type == 'lrm', \
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from x_ggrm.models import ModelLRM
        model = ModelLRM(**cfg.model, image_res=cfg.dataset.input_image_res)
        return model

    def _build_optimizer(self, model: nn.Module, cfg):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )

        return optimizer

    def _build_dataloader(self, cfg):
        from x_ggrm.datasets import CTLRMDataset, CTLRMDatasetCfg
        train_cfg = CTLRMDatasetCfg(**cfg.dataset.subsets,
                                    nums_input_view=cfg.dataset.nums_input_view,
                                    nums_target_views=cfg.dataset.nums_target_view,
                                    input_image_res=cfg.dataset.input_image_res,
                                    render_region_size=cfg.dataset.render_image.region,
                                    render_image_res_low=cfg.dataset.render_image.low,
                                    render_image_res_high=cfg.dataset.render_image.high,
                                    split='train',
                                    device='cpu')

        val_cfg = CTLRMDatasetCfg(**cfg.dataset.subsets,
                                  nums_input_view=cfg.dataset.nums_input_view,
                                  nums_target_views=cfg.dataset.nums_target_view,
                                  input_image_res=cfg.dataset.input_image_res,
                                  render_region_size=cfg.dataset.render_image.region,
                                  render_image_res_low=cfg.dataset.render_image.low,
                                  render_image_res_high=cfg.dataset.render_image.high,
                                  split='val',
                                  device='cpu')

        train_dataset = CTLRMDataset(train_cfg)
        val_dataset = CTLRMDataset(val_cfg)


        # build data loader
        # 优化：DataLoader配置优化
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.dataset.num_train_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.dataset.num_val_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=False
        )

        return train_loader, val_loader

    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        if cfg.train.scheduler.type == 'cosine':
            from x_ggrm.utils.scheduler_utils import CosineWarmupScheduler
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler

    def _compute_ct_volume_loss(self, vol_pred, vol_gt):
        """
        CT专用体积损失 - 现在vol_pred和vol_gt都在[0,1]范围内
        
        Args:
            vol_pred: 预测体积 [0,1] 范围 (sigmoid激活后)
            vol_gt: 真实体积 [0,1] 范围 (归一化后)
        """
        # 优化：减少调试信息频率，每500步输出一次即可
        if self.global_step % 500 == 0:
            logger.debug(f"vol_pred range: [{vol_pred.min():.4f}, {vol_pred.max():.4f}]")
            logger.debug(f"vol_gt range: [{vol_gt.min():.4f}, {vol_gt.max():.4f}]")
        
        loss = torch.nn.functional.mse_loss(vol_pred, vol_gt)
            
        return loss


    def _build_loss_fn(self, cfg):
        from x_ggrm.losses import PixelLoss, TVLoss
        pixel_loss_fn = PixelLoss()
        tv_loss_fn = TVLoss()
        return pixel_loss_fn, tv_loss_fn

    def register_hooks(self):
        pass

    def forward_loss_local_step(self, data):
        image_input = data['input_views']
        render_camera = data['render_cameras']
        render_image = data['render_images']
        render_anchors = data['render_anchors']
        render_full_resolutions = data['render_full_resolutions']
        vol_gt = data['vol_gt']

        # 优化：减少调试代码，只在必要时检查NaN（每1000步检查一次）
        if self.global_step % 1000 == 0 and torch.isnan(render_camera).any():
            logger.warning(f"[DATA WARNING] NaN detected in render_camera at step {self.global_step}")

        # forward
        outputs = self.model(
            image=image_input,
            render_cameras=render_camera,
            render_anchors=render_anchors,
            render_resolutions=render_full_resolutions,
            render_region_size=self.cfg.dataset.render_image.region,
        )

        # extract vol_pred from model output
        vol_pred = outputs['vol_pred']

        # loss calculation with improved weighting and normalization
        loss = 0.
        loss_pixel = None
        loss_tv = None
        loss_vol_mse = None

        # Progressive loss weighting for better convergence
        # Reduce regularization losses in early training to let the model learn basic structure first
        training_progress = min(self.global_step / max(1, self.N_max_global_steps * 0.3), 1.0)
        regularization_multiplier = training_progress

        if self.cfg.train.loss.pixel_weight > 0.:
            loss_pixel = self.pixel_loss_fn(outputs['images_density'], render_image)
            loss += loss_pixel * self.cfg.train.loss.pixel_weight
        if self.cfg.train.loss.tv_weight > 0.:
            loss_tv = self.tv_loss_fn(outputs['planes'])
            # Reduce TV loss weight in early training
            tv_weight = self.cfg.train.loss.tv_weight * regularization_multiplier
            loss += loss_tv * tv_weight
        if hasattr(self.cfg.train.loss, 'vol_mse_weight') and self.cfg.train.loss.vol_mse_weight > 0.:
            loss_vol_mse = self._compute_ct_volume_loss(vol_pred, vol_gt)
            # CT-aware volume loss with sparsity and structure preservation
            loss += loss_vol_mse * self.cfg.train.loss.vol_mse_weight

        return outputs, loss, loss_pixel, loss_tv, loss_vol_mse

    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        for data in loader:

            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):

                # forward to loss
                outs, loss, loss_pixel, loss_tv, loss_vol_mse = self.forward_loss_local_step(data)

                # backward
                self.accelerator.backward(loss)
                
                # 优化：减少梯度监控频率，只在必要时输出
                if self.accelerator.sync_gradients:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                    if self.global_step % 500 == 0:  # 减少到每500步检查一次
                        if grad_norm > 100 or grad_norm < 1e-6:
                            logger.warning(f"Gradient norm: {grad_norm:.4f} at step {self.global_step}")
                
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                local_step_losses.append(torch.stack([
                    _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                    for _loss in [loss, loss_pixel, loss_tv, loss_vol_mse]
                ]))

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                # 优化：移除不必要的调试日志
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                loss, loss_pixel, loss_tv, loss_vol_mse = global_step_loss.unbind()
                loss_kwargs = {
                    'loss': loss.item(),
                    'loss_pixel': loss_pixel.item(),
                    'loss_tv': loss_tv.item(),
                    'loss_vol_mse': loss_vol_mse.item() if not torch.isnan(loss_vol_mse) else 0.0,
                }
                self.log_scalar_kwargs(
                    step=self.global_step, split='train',
                    **loss_kwargs
                )
                self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
                local_step_losses = []
                global_step_losses.append(global_step_loss)

                # manage display
                pbar.update(1)
                description = {
                    **loss_kwargs,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                description = '[TRAIN STEP]' + \
                              ', '.join(
                                  f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
                pbar.set_description(description)

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                if 'images_density' in outs and self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
                    self.log_image_monitor(
                        step=self.global_step, split='train',
                        renders=outs['images_density'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=data['render_images'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )
                # 优化：减少vol_pred保存频率，每1000步保存一次
                if 'vol_pred' in outs and self.global_step % 1000 == 0:
                    self._save_vol_pred_monitoring(outs, data)

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_loss_pixel, epoch_loss_tv, epoch_loss_vol_mse = epoch_losses.unbind()
        epoch_loss_dict = {
            'loss': epoch_loss.item(),
            'loss_pixel': epoch_loss_pixel.item(),
            'loss_tv': epoch_loss_tv.item(),
            'loss_vol_mse': epoch_loss_vol_mse.item() if not torch.isnan(epoch_loss_vol_mse) else 0.0,
        }
        self.log_scalar_kwargs(
            epoch=self.current_epoch, split='train',
            **epoch_loss_dict,
        )
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
            ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):

        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        with tqdm(
                range(0, self.N_max_global_steps),
                initial=self.global_step,
                disable=(not self.accelerator.is_main_process),
        ) as pbar:

            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=10, warmup=10, active=100,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                    self.cfg.logger.tracker_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                )),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) if self.cfg.logger.enable_profiler else DummyProfiler()

            with profiler:

                self.optimizer.zero_grad()
                for _ in range(self.current_epoch, self.cfg.train.epochs):

                    loader = skipped_loader or self.train_loader
                    skipped_loader = None
                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)
                    if self.accelerator.check_trigger():
                        break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

            # final checkpoint and evaluation
            self.save_checkpoint()
            self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: int = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        sample_data, sample_outs = None, None

        for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):

            if len(running_losses) >= max_val_batches:
                logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            outs, loss, loss_pixel, loss_tv, loss_vol_mse = self.forward_loss_local_step(data)
            sample_data, sample_outs = data, outs

            running_losses.append(torch.stack([
                _loss if _loss is not None else torch.tensor(float('nan'), device=self.device)
                for _loss in [loss, loss_pixel, loss_tv, loss_vol_mse]
            ]))

        total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
        total_loss, total_loss_pixel, total_loss_tv, total_loss_vol_mse = total_losses.unbind()
        total_loss_dict = {
            'loss': total_loss.item(),
            'loss_pixel': total_loss_pixel.item(),
            'loss_tv': total_loss_tv.item(),
            'loss_vol_mse': total_loss_vol_mse.item() if not torch.isnan(total_loss_vol_mse) else 0.0,
        }

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            if 'images_density' in sample_outs:
                self.log_image_monitor(
                    epoch=epoch, split='val',
                    renders=sample_outs['images_density'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    gts=sample_data['render_images'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                )
        else:
            self.log_scalar_kwargs(
                step=self.global_step, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            if 'images_density' in sample_outs:
                self.log_image_monitor(
                    step=self.global_step, split='val',
                    renders=sample_outs['images_density'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    gts=sample_data['render_images'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                )

        return total_loss_dict

    @Trainer.control('on_main_process')
    def log_image_monitor(
            self, epoch: int = None, step: int = None, split: str = None,
            renders: torch.Tensor = None, gts: torch.Tensor = None,
    ):
        """Log images to TensorBoard with proper formatting
        
        Args:
            renders: Rendered images [N, M, C, H, W]
            gts: Ground truth images [N, M, C, H, W]
        """
        if renders is None or gts is None:
            logger.warning("Cannot log images: renders or gts is None")
            return
            
        try:
            # Ensure tensors are on CPU and in proper format
            renders = renders.detach().cpu().float()
            gts = gts.detach().cpu().float()
            
            # Handle potential shape mismatch between renders and gts
            if renders.shape != gts.shape:
                logger.debug(f"Shape mismatch detected: renders {renders.shape} vs gts {gts.shape}")
                # If gts has shape [N, M, H, W, 1], convert to [N, M, 1, H, W]
                if len(gts.shape) == 5 and gts.shape[-1] == 1:
                    gts = gts.permute(0, 1, 4, 2, 3)  # [N, M, H, W, 1] -> [N, M, 1, H, W]
                    logger.debug(f"Converted gts shape to: {gts.shape}")
                elif len(gts.shape) == 4 and gts.shape[-1] == 1:
                    gts = gts.permute(0, 3, 1, 2)  # [N, H, W, 1] -> [N, 1, H, W] 
                    logger.debug(f"Converted gts shape to: {gts.shape}")
                
                if renders.shape != gts.shape:
                    logger.warning(f"Unable to resolve shape mismatch: renders {renders.shape} vs gts {gts.shape}")
                    return
            
            # Get dimensions
            if len(renders.shape) == 5:
                N, M, C, H, W = renders.shape
            elif len(renders.shape) == 4:
                # Handle case where M dimension might be missing
                N, C, H, W = renders.shape
                M = 1
                renders = renders.unsqueeze(1)  # [N, 1, C, H, W]
                gts = gts.unsqueeze(1)
            else:
                logger.warning(f"Unsupported tensor shape: {renders.shape}")
                return
            
            # Normalize to [0, 1] range with proper handling of different value ranges
            # Check if images are in [-1, 1] range and convert to [0, 1]
            if renders.min() < 0:
                renders = (renders + 1.0) / 2.0
            if gts.min() < 0:
                gts = (gts + 1.0) / 2.0
                
            # Final clamp to ensure [0, 1] range
            renders = torch.clamp(renders, 0.0, 1.0)
            gts = torch.clamp(gts, 0.0, 1.0)
            
            # Handle grayscale images by converting to RGB for better visualization
            if C == 1:
                renders = renders.repeat(1, 1, 3, 1, 1)  # [N, M, 3, H, W]
                gts = gts.repeat(1, 1, 3, 1, 1)  # [N, M, 3, H, W]
                C = 3
            elif C != 3:
                logger.warning(f"Unsupported number of channels: {C}, skipping image logging")
                return
            
            # Limit the number of images to prevent memory issues
            max_images = min(N, 4)  # Show at most 4 batch items
            renders = renders[:max_images]
            gts = gts[:max_images]
            N = max_images
            
            # Reshape for grid creation: [N*M, C, H, W]
            renders_flat = renders.view(-1, C, H, W)
            gts_flat = gts.view(-1, C, H, W)
            
            # Create image grids with proper spacing
            renders_grid = make_grid(
                renders_flat, 
                nrow=M, 
                normalize=False, 
                value_range=(0, 1),
                pad_value=1.0  # White padding for better separation
            )
            gts_grid = make_grid(
                gts_flat, 
                nrow=M, 
                normalize=False, 
                value_range=(0, 1),
                pad_value=1.0
            )
            
            # Create side-by-side comparison grid
            comparison_images = []
            for i in range(N):
                for j in range(M):
                    comparison_images.append(renders[i, j])  # Rendered image
                    comparison_images.append(gts[i, j])      # GT image
            
            if comparison_images:
                comparison_flat = torch.stack(comparison_images, dim=0)  # [N*M*2, C, H, W]
                comparison_grid = make_grid(
                    comparison_flat, 
                    nrow=M*2, 
                    normalize=False, 
                    value_range=(0, 1),
                    pad_value=1.0
                )
            else:
                comparison_grid = renders_grid  # Fallback
            
            # Get logging progress
            log_type, log_progress = self._get_str_progress(epoch, step)
            split = f'/{split}' if split else ''
            
            # Ensure grids are in proper format for TensorBoard [1, C, H, W]
            # TensorBoard expects 4D tensors in NCHW format
            if len(renders_grid.shape) == 3:
                renders_grid = renders_grid.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            if len(gts_grid.shape) == 3:
                gts_grid = gts_grid.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            if len(comparison_grid.shape) == 3:
                comparison_grid = comparison_grid.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            
            # Log images to TensorBoard
            self.log_images({
                f'Images{split}/rendered': renders_grid,
                f'Images{split}/ground_truth': gts_grid,
                f'Images{split}/comparison': comparison_grid,
            }, log_progress)
            
            logger.debug(f"Successfully logged images to TensorBoard: {split} at {log_type} {log_progress}")
            
        except Exception as e:
            logger.error(f"Error logging images to TensorBoard: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    @Trainer.control('on_main_process')
    def _save_vol_pred_monitoring(self, outs, data):
        """Save vol_pred and vol_gt for monitoring training progress"""
        try:
            import numpy as np
            
            # Create monitoring directory
            monitor_dir = os.path.join('./vol_monitoring', 
                                     self.cfg.experiment.parent, 
                                     self.cfg.experiment.child)
            os.makedirs(monitor_dir, exist_ok=True)
            
            # Get vol_pred and vol_gt
            vol_pred = outs['vol_pred'].detach().cpu().numpy()  # [N, D, H, W]
            vol_gt = data['vol_gt'].detach().cpu().numpy()      # [N, D, H, W]
            
            # Save data
            step_file = os.path.join(monitor_dir, f'step_{self.global_step:06d}.npz')
            np.savez_compressed(step_file,
                              vol_pred=vol_pred[0],      # Save first batch item
                              vol_gt=vol_gt[0],          # Save first batch item
                              step=self.global_step)
            
            # Log statistics
            logger.info(f"[Vol Monitor] Step {self.global_step}:")
            logger.info(f"  vol_pred: min={vol_pred.min():.6f}, max={vol_pred.max():.6f}, mean={vol_pred.mean():.6f}")
            logger.info(f"  vol_gt:   min={vol_gt.min():.6f}, max={vol_gt.max():.6f}, mean={vol_gt.mean():.6f}")
            logger.info(f"  MSE loss: {((vol_pred - vol_gt)**2).mean():.6f}")
            logger.info(f"  Saved to: {step_file}")
            
        except Exception as e:
            logger.error(f"Error saving vol_pred monitoring: {e}")
            import traceback
            logger.debug(traceback.format_exc())
