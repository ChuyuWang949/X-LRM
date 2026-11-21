"""Export one fixed-view LNDb case using the native X-LRM input adapter."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import SimpleITK as sitk
import torch

from x_ggrm.datasets.lndb_lrm import LNDbLRMDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--case', default='LNDb-0045')
    parser.add_argument('--views', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/train-lrm-lndb-distill.yaml')
    cfg.dataset.lndb_root = str(args.dataset_root)
    cfg.dataset.nums_input_view = args.views
    cfg.model.volume_grid_size = 256
    cfg.distillation.weight = 0
    splits = json.loads((args.dataset_root / 'splits.json').read_text())
    split, = [name for name, cases in splits.items() if args.case in cases]
    dataset = LNDbLRMDataset(cfg, split)
    dataset.training = False
    record = dataset._read_projections(dataset.names.index(args.case))
    if not 0 < args.views <= len(record['angles']):
        raise ValueError('View count must be within the acquired frame count')
    frames = dataset._view_indices(len(record['angles']))
    inputs, _, poses = dataset._encode_projections(record, frames)
    raw = np.ascontiguousarray(record['projections'][frames])
    angles = np.asarray(record['angles'][frames])
    ct = sitk.GetArrayFromImage(sitk.ReadImage(
        str(args.dataset_root / 'processed' / f'{args.case}.nii.gz')))
    assert ct.dtype == np.uint8 and list(ct.shape) == dataset.geometry['nVoxel']
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'volume.npz', ct_xyz=ct)
    np.savez_compressed(output / 'projections.npz', projections=raw, angles_rad=angles,
                        source_frame_indices=frames)
    np.savez_compressed(output / 'input_views.npz', input_views=inputs.numpy())
    # Preserve the acquisition configuration, including its noise settings.
    (output / 'acquisition.yaml').write_bytes((args.dataset_root / 'config.yaml').read_bytes())

    scale = dataset.units_per_metre
    config = dataset.geometry
    k = dataset.intrinsic.numpy()
    height, width = config['nDetector']
    detector_height, detector_width = config['sDetector']
    cameras = []
    max_inverse_error = 0.
    max_ray_error = 0.
    for index, source_frame in enumerate(frames):
        c2w_normalized = poses[index].numpy().astype(np.float64)
        c2w_m = c2w_normalized.copy()
        c2w_m[:3, 3] /= scale
        c2w_cv_m = c2w_m @ np.diag([1., -1., -1., 1.])
        w2c_m = np.linalg.inv(c2w_m)
        np.testing.assert_allclose(c2w_m[:3, :3].T @ c2w_m[:3, :3], np.eye(3), atol=2e-7)
        np.testing.assert_allclose(np.linalg.det(c2w_m[:3, :3]), 1., atol=2e-7)
        max_inverse_error = max(max_inverse_error, float(np.abs(c2w_m @ w2c_m - np.eye(4)).max()))
        source = c2w_m[:3, 3]
        forward = c2w_cv_m[:3, 2]
        u_axis = c2w_cv_m[:3, 0]
        v_axis = c2w_cv_m[:3, 1]
        detector_center = source + config['DSD'] * forward
        top_left_center = (detector_center - detector_width / 2 * u_axis
                           - detector_height / 2 * v_axis
                           + detector_width / width / 2 * u_axis
                           + detector_height / height / 2 * v_axis)
        size = dataset.image_size
        for row, col in [(0, 0), (size // 2, size // 2), (size - 1, size - 1)]:
            target = (detector_center + ((col + .5) / size - .5) * detector_width * u_axis
                      + ((row + .5) / size - .5) * detector_height * v_axis)
            direction = target - source
            direction /= np.linalg.norm(direction)
            max_ray_error = max(max_ray_error, float(np.abs(direction - inputs[index, row, col, 4:7].numpy()).max()))
        cameras.append(dict(
            frame_index=index, source_frame_index=int(source_frame),
            angle_rad=float(angles[index]), angle_deg=float(np.degrees(angles[index])),
            world_from_camera_blender_m=c2w_m.tolist(),
            camera_from_world_blender_m=w2c_m.tolist(),
            world_from_camera_opencv_m=c2w_cv_m.tolist(),
            camera_from_world_opencv_m=np.linalg.inv(c2w_cv_m).tolist(),
            world_from_camera_blender_model=c2w_normalized.tolist(),
            source_m=source.tolist(), detector_center_m=detector_center.tolist(),
            detector_u_axis_world=u_axis.tolist(), detector_v_axis_world=v_axis.tolist(),
            detector_top_left_pixel_center_m=top_left_center.tolist()))

    geometry = dict(
        length_unit='m', angle_unit='rad',
        acquisition_lengths_mm={
            'DSO': config['DSO'] * 1000, 'DSD': config['DSD'] * 1000,
            'volume_extent_xyz': (np.asarray(config['sVoxel']) * 1000).tolist(),
            'voxel_spacing_xyz': (np.asarray(config['sVoxel']) / config['nVoxel'] * 1000).tolist(),
            'detector_extent_vu': (np.asarray(config['sDetector']) * 1000).tolist(),
            'detector_spacing_vu': (np.asarray(config['sDetector']) / config['nDetector'] * 1000).tolist()},
        ct_axes='XYZ, matching SimpleITK.GetArrayFromImage on the source file',
        ct_dtype='uint8', ct_to_model_intensity='ct_xyz.astype(float32) / 255',
        ct_physical_geometry='acquisition.yaml; source NIfTI header is an index-space container',
        model_point_queries='q_xyz = 2 * index_xyz / (nVoxel_xyz - 1) - 1',
        acquisition_voxel_centers_m='(index_xyz + 0.5) * sVoxel_xyz / nVoxel_xyz - sVoxel_xyz / 2',
        model_query_note='The endpoint query lattice and acquisition voxel-center lattice are recorded separately.',
        model_length_units_per_metre=scale,
        model_intrinsics_normalized=k.tolist(),
        image_pixel_coordinates='u=(column+0.5)/width, v=(row+0.5)/height',
        camera_convention='Blender camera +X right, +Y up, -Z forward; OpenCV +X right, +Y down, +Z forward',
        pose_convention='world_from_camera maps camera points into the centered CT world',
        detector_array_axes='view,row(v),column(u)',
        input_channels=['projection_integral_model', 'source_x_model', 'source_y_model',
                        'source_z_model', 'ray_direction_x_world', 'ray_direction_y_world',
                        'ray_direction_z_world'],
        input_shape=list(inputs.shape), ray_encoding='origin_direction',
        projection_processing=dict(
            source='archived float32 CT-derived projections',
            raw_negative_values='preserved in projections.npz',
            model_intensity='maximum(raw_projection, 0) * model_length_units_per_metre',
            resize='PyTorch bilinear, align_corners=False, antialias=True, 512x512 to 224x224',
            additional_noise_applied=False,
            stored_acquisition_noise='See acquisition.yaml; exporting reuses archived pixels.'),
        cameras=cameras)
    (output / 'geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')

    image_scale = float(np.maximum(raw, 0).max() * scale)
    montage = Image.new('RGB', (6 * 256, ((args.views + 5) // 6) * 280), 'white')
    draw = ImageDraw.Draw(montage)
    with (output / 'frames.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['frame_index', 'source_frame_index', 'angle_rad', 'angle_deg', 'detector_png', 'input_png'])
        for i, source_frame in enumerate(frames):
            stem = f'frame_{i:02d}_source_{source_frame:03d}.png'
            for folder, image_values in [
                    ('detector_png', np.maximum(raw[i], 0) * scale),
                    ('input_png', inputs[i, :, :, 0].numpy())]:
                (output / folder).mkdir(exist_ok=True)
                pixels = np.round(np.clip(image_values / image_scale, 0, 1) * 65535).astype(np.uint16)
                Image.fromarray(pixels).save(output / folder / stem)
            thumbnail = Image.fromarray(np.round(np.clip(inputs[i, :, :, 0].numpy() / image_scale, 0, 1) * 255).astype(np.uint8))
            x, y = i % 6 * 256, i // 6 * 280
            montage.paste(thumbnail.resize((256, 256)), (x, y + 24))
            draw.text((x + 4, y + 5), f'{i:02d} / source {source_frame:02d} / {np.degrees(angles[i]):.1f} deg', fill='black')
            writer.writerow([i, source_frame, repr(float(angles[i])), repr(float(np.degrees(angles[i]))),
                             f'detector_png/{stem}', f'input_png/{stem}'])
    montage.save(output / 'projection_montage.png')

    np.testing.assert_array_equal(np.load(output / 'volume.npz')['ct_xyz'], ct)
    np.testing.assert_array_equal(np.load(output / 'projections.npz')['projections'], raw)
    np.testing.assert_array_equal(np.load(output / 'input_views.npz')['input_views'], inputs.numpy())
    np.testing.assert_allclose(np.linalg.norm(inputs[..., 4:7].numpy(), axis=-1), 1, atol=2e-7)
    np.testing.assert_allclose(inputs[..., 1:4].numpy(),
                               np.broadcast_to(poses[:, None, None, :3, 3].numpy(), inputs[..., 1:4].shape))
    assert max_ray_error < 1e-6 and max_inverse_error < 1e-12
    metadata = dict(
        case=args.case, source_split=split, views=args.views,
        source_acquired_frames=len(record['angles']), source_frame_indices=frames.tolist(),
        source_selection='floor(linspace(0, source_acquired_frames, views, endpoint=False))',
        raw_projection_shape=list(raw.shape), volume_shape=list(ct.shape),
        model_input_shape=list(inputs.shape), png_encoding='uint16 PNG',
        png_decode_model_integral_scale=image_scale / 65535,
        png_note='PNG previews clamp archived negative values; floating-point NPZ arrays are the quantitative inputs.',
        qa=dict(exact_volume_roundtrip=True, exact_projection_roundtrip=True, exact_input_roundtrip=True,
                rigid_camera_checks_passed=True, max_camera_inverse_error=max_inverse_error,
                max_ray_direction_component_error=max_ray_error))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
