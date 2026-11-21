"""LNDb DIF-Net-format projections in the native XLRM input layout."""

import json
from pathlib import Path
import pickle

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml


class LNDbLRMDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.root = Path(cfg.dataset.lndb_root)
        self.training = split == 'train'
        splits = json.loads((self.root / 'splits.json').read_text())
        self.names = splits[split]
        if any(set(splits[a]) & set(splits[b]) for a,b in
               [('train', 'eval'), ('train', 'test'), ('eval', 'test')]):
            raise ValueError('LNDb patient lists overlap')
        for case in self.names:
            if not (self.root/'processed'/f'{case}.nii.gz').is_file() or not (
                    self.root/'projections'/case/'all.pickle').is_file():
                raise FileNotFoundError(case)
        self.geometry = g = yaml.safe_load((self.root/'config.yaml').read_text())
        extent = np.asarray(g['sVoxel'])
        if (g['mode'] != 'cone' or not np.allclose(extent, extent[0])
                or any(g['offOrigin']) or any(g['offDetector']) or not 0 < g['DSO'] < g['DSD'] < 10):
            raise ValueError('Expected LNDb centered cubic geometry with lengths in metres')
        self.units_per_metre = 2.0 / extent[0]
        self.views = cfg.dataset.nums_input_view
        self.training_views = tuple(cfg.dataset.get('training_view_counts', ())) if self.training else ()
        if self.training_views and (
            cfg.train.batch_size != 1 or any(not 0 < v <= g['numTrain'] for v in self.training_views)
        ):
            raise ValueError('Mixed view counts need batch size 1 and valid acquisition counts')
        self.ray_encoding = cfg.dataset.get('ray_encoding', 'origin_direction')
        if self.ray_encoding not in ('origin_direction', 'rppc'):
            raise ValueError(f'Unknown LNDb ray encoding: {self.ray_encoding}')
        distillation = cfg.get('distillation')
        self.teacher_views = (
            int(distillation.teacher_views)
            if self.training and distillation and distillation.weight > 0
            else 0
        )
        if self.teacher_views and not self.views < self.teacher_views <= g['numTrain']:
            raise ValueError('Expected sparse views < teacher views <= acquired views')
        if self.teacher_views and self.training_views:
            raise ValueError('Mixed view counts are not supported with distillation')
        self.target_views = cfg.dataset.nums_target_view
        self.image_size = cfg.dataset.input_image_res
        self.render_size = cfg.dataset.render_image.high
        self.region = cfg.dataset.render_image.region
        self.volume_size = cfg.model.volume_grid_size
        self.ct_points_per_case = int(cfg.train.get('ct_points_per_case', 0))
        self.ct_volume_only = bool(cfg.train.get('ct_volume_only', False))
        if self.training and self.ct_points_per_case:
            self.blocks = np.load(self.root / 'blocks' / 'blocks.npy', mmap_mode='r', allow_pickle=False)
            if not 0 < self.ct_points_per_case <= self.blocks.shape[1]:
                raise ValueError('CT point count must fit within one LNDb block')
        detector_height, detector_width = g['sDetector']
        self.intrinsic = torch.tensor([[g['DSD']/detector_width, 0, .5],
                                       [0, g['DSD']/detector_height, .5],
                                       [0, 0, 1]], dtype=torch.float32)
        u = (torch.arange(self.image_size) + .5) / self.image_size - .5
        y, x = torch.meshgrid(u, u, indexing='ij')
        self.pixel_rays = torch.stack((x / self.intrinsic[0, 0], y / self.intrinsic[1, 1],
                                       torch.ones_like(x)), dim=-1)

    def __len__(self):
        return len(self.names) * (len(self.training_views) or 1)

    def cameras(self, angles):
        c, s = angles.cos(), angles.sin()
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        radial = torch.stack((c, s, zero), dim=-1)
        right = torch.stack((-s, c, zero), dim=-1)
        up = torch.stack((zero, zero, one), dim=-1)
        pose = torch.eye(4).repeat(len(c), 1, 1)
        # RaySampler converts the renderer pose from Blender to OpenCV.
        pose[:, :3, :3] = torch.stack((right, up, radial), dim=-1)
        pose[:, :3, 3] = radial * (self.geometry['DSO'] * self.units_per_metre)
        return pose

    def _read_projections(self, index):
        case = self.names[index]
        with (self.root/'projections'/case/'all.pickle').open('rb') as stream:
            record = pickle.load(stream)
        raw = record['projections']
        g = self.geometry
        if raw.shape != (g['numTrain'], *g['nDetector']) or len(record['angles']) != len(raw):
            raise ValueError(f'Projection geometry mismatch for {case}')
        return record

    def _view_indices(self, n_acquired, views=None):
        ids = np.linspace(0, n_acquired, self.views if views is None else views, endpoint=False).astype(int)
        if self.training:
            ids += np.random.randint(n_acquired - ids[-1])
        return ids

    def _encode_projections(self, record, ids):
        raw = record['projections']
        angles = torch.from_numpy(np.asarray(record['angles'][ids], dtype=np.float32))
        # Existing TIGRE projections integrate normalized density in metres.
        # XLRM integrates over [-1, 1]^3, so projection integrals scale with length.
        images = torch.from_numpy(np.maximum(raw[ids], 0).astype(np.float32))[:, None]
        images *= self.units_per_metre
        projection_inputs = F.interpolate(images, (self.image_size,)*2, mode='bilinear',
                                          align_corners=False, antialias=True)
        pose = self.cameras(angles)
        opencv_rotation = pose[:, :3, :3] * torch.tensor([1., -1., -1.])
        rays = F.normalize(torch.einsum('vij,hwj->vhwi', opencv_rotation, self.pixel_rays), dim=-1)
        origins = pose[:, None, None, :3, 3].expand_as(rays)
        ray_positions = origins
        if self.ray_encoding == 'rppc':
            # Reference point is the orthogonal projection of isocenter onto the ray.
            ray_positions = origins - (origins * rays).sum(dim=-1, keepdim=True) * rays
        inputs = torch.cat((projection_inputs.permute(0, 2, 3, 1), ray_positions, rays), dim=-1)

        return inputs, images, pose

    def projection_inputs(self, index, views=None):
        record = self._read_projections(index)
        ids = self._view_indices(len(record['angles']), views)
        return self._encode_projections(record, ids)

    def _load_volume(self, case):
        # The shared NIfTI stores the legacy array directly in XYZ order.
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(self.root/'processed'/f'{case}.nii.gz')))
        if vol.dtype != np.uint8 or list(vol.shape) != self.geometry['nVoxel']:
            raise ValueError(f'Unexpected shared LNDb CT array for {case}')
        volume = torch.from_numpy(vol.astype(np.float32)) / 255
        if self.volume_size != volume.shape[0]:
            volume = F.interpolate(volume[None,None], size=(self.volume_size,)*3,
                                   mode='trilinear', align_corners=True)[0,0]
        return volume

    def _sample_ct_points(self, case):
        block_index = np.random.randint(len(self.blocks))
        values = np.load(self.root / 'blocks' / case / f'block_{block_index}.npy', allow_pickle=False)
        coords = self.blocks[block_index]
        selected = np.random.choice(len(coords), size=self.ct_points_per_case, replace=False)
        return dict(ct_query_points=torch.from_numpy(coords[selected].astype(np.float32)) * 2 - 1,
                    ct_point_targets=torch.from_numpy(values[selected].astype(np.float32)) / 255)

    def __getitem__(self, index):
        views = None
        if self.training_views:
            index, view_index = divmod(index, len(self.training_views))
            views = self.training_views[view_index]
        case = self.names[index]
        teacher = {}
        if self.teacher_views:
            record = self._read_projections(index)
            count = len(record['angles'])
            ids = self._view_indices(count)
            remaining = np.setdiff1d(np.arange(count), ids, assume_unique=True)
            extra_count = self.teacher_views - len(ids)
            extra = remaining[((np.arange(extra_count) + .5) * len(remaining) / extra_count).astype(int)]
            dense_ids = np.sort(np.concatenate((ids, extra)))
            dense_inputs, dense_images, dense_pose = self._encode_projections(record, dense_ids)
            positions = np.searchsorted(dense_ids, ids)
            inputs, images, pose = dense_inputs[positions], dense_images[positions], dense_pose[positions]
            teacher = dict(teacher_input_views=dense_inputs, input_view_indices=torch.from_numpy(ids),
                           teacher_view_indices=torch.from_numpy(dense_ids))
        else:
            inputs, images, pose = self.projection_inputs(index, views)

        if self.ct_points_per_case:
            supervision = self._sample_ct_points(case) if self.training else {'vol_gt': self._load_volume(case)}
            return dict(id=case, input_views=inputs, **supervision, **teacher)
        if self.ct_volume_only:
            return dict(id=case, input_views=inputs, vol_gt=self._load_volume(case), **teacher)

        ids_target = (np.sort(np.random.choice(self.views, self.target_views, replace=False))
                      if self.training else
                      np.linspace(0, self.views - 1, self.target_views).round().astype(int))
        targets = F.interpolate(images[ids_target], (self.render_size,)*2, mode='bilinear',
                                align_corners=False, antialias=True)
        if self.training:
            anchors = torch.randint(0, self.render_size - self.region + 1, (self.target_views, 2))
        else:
            anchors = torch.full((self.target_views, 2), (self.render_size - self.region) // 2)
        crops = torch.stack([image[:, y:y+self.region, x:x+self.region]
                             for image, (y, x) in zip(targets, anchors.tolist())])
        cameras = torch.cat((pose[ids_target].flatten(1),
                             self.intrinsic.flatten()[None].expand(self.target_views, -1)), dim=1)
        volume = self._load_volume(case)
        return dict(id=case, input_views=inputs, render_cameras=cameras,
                    render_images=crops, render_anchors=anchors,
                    render_full_resolutions=torch.full((self.target_views, 1), self.render_size),
                    vol_gt=volume, **teacher)
