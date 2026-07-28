# Data adapter for D:/batch_organized luna.organized_sequence data.

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from loguru import logger as logging
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class LunaOrganized(data.Dataset):
    """Dataset adapter for luna.organized_sequence stereo data.

    The adapter reads left/right RGB images and dense uint16 depth PNGs, then
    projects the selected LiDAR source into the left image to build sparse
    guided disparity hints.
    """

    def __init__(self, aug_params=None, root='D:/batch_organized', image_set='training', args=None):
        self.root = Path(root)
        assert self.root.exists(), self.root

        self.args = args
        self.sparse = True
        self.kitti_completion = True
        self.is_test = False
        self.init_seed = False
        self.img_pad = aug_params.pop('img_pad', None) if aug_params is not None else None
        self.augmentor = None
        if aug_params is not None and 'crop_size' in aug_params:
            self.augmentor = SparseFlowAugmentor(args, **aug_params)

        self.resize_hw = getattr(args, 'luna_resize', None)
        if self.resize_hw is not None and len(self.resize_hw) == 0:
            self.resize_hw = None
        self.depth_scale = float(getattr(args, 'luna_depth_scale', 1.0 / 256.0))
        self.max_time_diff_ns = int(getattr(args, 'luna_lidar_max_time_diff_ns', 100000000))
        self.require_lidar = bool(getattr(args, 'luna_require_lidar', 1))
        self.camera_key = getattr(args, 'luna_camera_key', 'Cam_Rect_L')
        self.apply_rectification = bool(getattr(args, 'luna_apply_rectification', 1))
        self.image_subdir = Path(getattr(args, 'luna_image_subdir', 'images'))
        self.left_dirname = getattr(args, 'luna_left_dirname', 'left')
        self.right_dirname = getattr(args, 'luna_right_dirname', 'right')
        self.depth_subdir = Path(getattr(args, 'luna_depth_subdir', 'depth_gt'))
        self.lidar_source = getattr(args, 'luna_lidar_source', 'raw')
        if self.lidar_source not in {'raw', 'fastlio', 'fake'}:
            raise ValueError(f'Unsupported Luna LiDAR source: {self.lidar_source}')
        body_to_lidar = getattr(args, 'luna_fastlio_body_to_lidar', None)
        self.fastlio_body_to_lidar = (
            np.eye(4, dtype=np.float64)
            if body_to_lidar is None
            else np.asarray(body_to_lidar, dtype=np.float64).reshape(4, 4)
        )
        self.border_crop_fraction = float(
            getattr(args, 'luna_border_crop_fraction', 0.0)
        )
        if not 0.0 <= self.border_crop_fraction < 0.5:
            raise ValueError(
                'luna_border_crop_fraction must satisfy 0 <= fraction < 0.5'
            )

        self.image_list = []
        self.disparity_list = []
        self.sparse_hint_list = []
        self.extra_info = []

        for sample in self._collect_samples(image_set):
            self.image_list.append([str(sample['left']), str(sample['right'])])
            self.disparity_list.append(str(sample['depth']))
            self.sparse_hint_list.append(str(sample['lidar']) if sample['lidar'] is not None else '')
            self.extra_info.append(sample)

    def __len__(self):
        return len(self.image_list)

    def __mul__(self, v):
        import copy
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.sparse_hint_list = v * copy_of_self.sparse_hint_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def _collect_samples(self, image_set):
        all_samples = []
        skipped_no_lidar = 0
        excluded_sequences = set(getattr(self.args, 'luna_exclude_sequences', []) or [])
        if excluded_sequences:
            logging.info(f'Excluding Luna sequences: {sorted(excluded_sequences)}')
        logging.info(
            f'Luna inputs: images={self.image_subdir}/'
            f'{self.left_dirname},{self.right_dirname}, depth={self.depth_subdir}, '
            f'lidar={self.lidar_source}, border_crop={self.border_crop_fraction:.1%}'
        )
        for seq in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            if seq.name in excluded_sequences:
                continue
            if not (seq / 'manifest.json').exists() and not (seq / 'calibration').exists():
                continue
            left_dir = seq / self.image_subdir / self.left_dirname
            right_dir = seq / self.image_subdir / self.right_dirname
            depth_dir = seq / self.depth_subdir
            lidar_root = seq / f'lidar_{self.lidar_source}'
            lidar_dir = lidar_root / 'frames'
            if not left_dir.exists() or not right_dir.exists() or not depth_dir.exists():
                logging.warning(f'Skipping {seq.name}: missing stereo image or depth directory')
                continue
            if self.require_lidar and not lidar_dir.exists():
                logging.warning(
                    f'Skipping {seq.name}: missing {self.lidar_source} LiDAR directory'
                )
                continue

            calib = self._read_calibration(seq / 'calibration')
            if self.lidar_source == 'fastlio':
                calib['trajectory'] = self._read_fastlio_trajectory(
                    lidar_root / 'trajectory.npz'
                )
            image_times = self._read_image_timestamps(seq / 'images' / 'timestamps.tsv')
            timestamp_name = (
                'timestamps.tsv' if self.lidar_source == 'fastlio'
                else 'timestamps.txt'
            )
            lidar_times = self._read_lidar_timestamps(
                lidar_root / timestamp_name
            ) if lidar_dir.exists() else []
            existing_lidar_times = [
                item for item in lidar_times
                if (lidar_root / item[1]).exists()
            ]
            missing_lidar_files = len(lidar_times) - len(existing_lidar_times)
            if missing_lidar_files:
                logging.warning(
                    f'Ignoring {missing_lidar_files} {self.lidar_source} timestamp '
                    f'entries without files in {seq.name}'
                )
            lidar_times = existing_lidar_times
            lidar_ns = [item[0] for item in lidar_times]

            common = sorted(
                {p.stem for p in left_dir.glob('*.png')} &
                {p.stem for p in right_dir.glob('*.png')} &
                {p.stem for p in depth_dir.glob('*.png')}
            )

            for stem in common:
                lidar_path = None
                image_ns = image_times.get(('left', stem))
                if lidar_times and image_ns is not None:
                    lidar_path = self._nearest_lidar(
                        lidar_root, image_ns, lidar_ns, lidar_times
                    )
                if self.require_lidar and lidar_path is None:
                    skipped_no_lidar += 1
                    continue
                all_samples.append({
                    'sequence': seq.name,
                    'frame': stem,
                    'left': left_dir / f'{stem}.png',
                    'right': right_dir / f'{stem}.png',
                    'depth': depth_dir / f'{stem}.png',
                    'lidar': lidar_path,
                    'calib': calib,
                    'image_time_ns': image_ns,
                })

        if skipped_no_lidar:
            logging.warning(
                f'Skipped {skipped_no_lidar} Luna samples without a matched '
                f'{self.lidar_source} LiDAR frame'
            )

        if image_set in ['training', 'val', 'test']:
            return self._split_samples(all_samples, image_set=image_set)
        return all_samples

    def _split_samples(self, samples, image_set):
        val_fraction = float(getattr(self.args, 'luna_val_fraction', 0.2))
        test_fraction_arg = getattr(self.args, 'luna_test_fraction', None)
        test_fraction = val_fraction if test_fraction_arg is None else float(test_fraction_arg)
        if len(samples) == 0:
            return []
        val_num = int(round(len(samples) * val_fraction)) if val_fraction > 0 else 0
        test_num = int(round(len(samples) * test_fraction)) if test_fraction > 0 else 0
        if val_fraction > 0:
            val_num = min(max(val_num, 1), len(samples))
        if test_fraction > 0:
            test_num = min(max(test_num, 1), len(samples) - val_num)
        if val_num + test_num >= len(samples):
            overflow = val_num + test_num - len(samples) + 1
            test_num = max(0, test_num - overflow)
        state = np.random.get_state()
        np.random.seed(1000)
        split_idxs = np.random.permutation(len(samples))
        np.random.set_state(state)
        val_idxs = set(split_idxs[:val_num])
        test_idxs = set(split_idxs[val_num:val_num + test_num])
        if image_set == 'val':
            return [sample for idx, sample in enumerate(samples) if idx in val_idxs]
        if image_set == 'test':
            return [sample for idx, sample in enumerate(samples) if idx in test_idxs]
        holdout_idxs = val_idxs | test_idxs
        return [sample for idx, sample in enumerate(samples) if idx not in holdout_idxs]
    def _read_calibration(self, calib_dir):
        intrinsics = json.load(open(calib_dir / 'intrinsics.json', 'r', encoding='utf-8'))
        extrinsics = json.load(open(calib_dir / 'extrinsics.json', 'r', encoding='utf-8'))
        camera = intrinsics[self.camera_key]
        k = np.array(camera['K'], dtype=np.float32)
        right_to_left = np.array(extrinsics['Cam_R_to_Cam_L']['T'], dtype=np.float32).reshape(3)
        baseline_m = float(np.linalg.norm(right_to_left) / 1000.0)
        lidar_to_cam = np.array(extrinsics['LiDAR_to_Cam_L']['RT'], dtype=np.float32)
        if self.apply_rectification and 'Cam_L_to_Cam_L_Rect' in extrinsics:
            rect_r = np.array(extrinsics['Cam_L_to_Cam_L_Rect']['R'], dtype=np.float32)
        else:
            rect_r = np.eye(3, dtype=np.float32)
        return {
            'K': k,
            'baseline_m': baseline_m,
            'conversion_rate': float(k[0, 0] * baseline_m),
            'lidar_to_cam': lidar_to_cam,
            'rect_r': rect_r,
        }

    def _read_image_timestamps(self, path):
        result = {}
        if not path.exists():
            return result
        with open(path, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    result[(parts[0], parts[1])] = int(parts[2])
        return result

    def _read_lidar_timestamps(self, path):
        result = []
        if not path.exists():
            return result
        with open(path, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    result.append((int(parts[2]), parts[1]))
        return sorted(result)

    def _nearest_lidar(self, lidar_root, image_ns, lidar_ns, lidar_times):
        pos = np.searchsorted(lidar_ns, image_ns)
        candidates = []
        if pos < len(lidar_times):
            candidates.append(lidar_times[pos])
        if pos > 0:
            candidates.append(lidar_times[pos - 1])
        if not candidates:
            return None
        best_ns, rel_path = min(candidates, key=lambda item: abs(item[0] - image_ns))
        if abs(best_ns - image_ns) > self.max_time_diff_ns:
            return None
        return lidar_root / rel_path

    def _read_fastlio_trajectory(self, path):
        if not path.exists():
            raise FileNotFoundError(f'Missing FAST-LIO trajectory: {path}')
        with np.load(path) as trajectory:
            times = trajectory['header_time_ns'].astype(np.int64)
            order = np.argsort(times)
            return {
                'header_time_ns': times[order],
                'position': trajectory['position'][order].astype(np.float64),
                'orientation': trajectory['orientation'][order].astype(np.float64),
            }

    def _interpolate_fastlio_pose(self, trajectory, timestamp_ns):
        times = trajectory['header_time_ns']
        positions = trajectory['position']
        quaternions = trajectory['orientation']
        relative_times = (times - times[0]) * 1e-9
        query_time = float(
            np.clip(
                (timestamp_ns - times[0]) * 1e-9,
                relative_times[0],
                relative_times[-1],
            )
        )
        upper = int(np.searchsorted(relative_times, query_time))
        if upper == 0:
            return positions[0], Rotation.from_quat(quaternions[0]).as_matrix()
        if upper == len(relative_times):
            return positions[-1], Rotation.from_quat(quaternions[-1]).as_matrix()

        lower = upper - 1
        alpha = (query_time - relative_times[lower]) / (
            relative_times[upper] - relative_times[lower]
        )
        position = (1.0 - alpha) * positions[lower] + alpha * positions[upper]
        rotation = Slerp(
            relative_times[[lower, upper]],
            Rotation.from_quat(quaternions[[lower, upper]]),
        )([query_time]).as_matrix()[0]
        return position, rotation

    def _read_pcd_xyz(self, filename):
        fields, sizes, types, counts = None, None, None, None
        points, data_type = None, None
        header_len = 0
        with open(filename, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                header_len += len(line)
                text = line.decode('ascii', errors='ignore').strip()
                if text.startswith('FIELDS'):
                    fields = text.split()[1:]
                elif text.startswith('SIZE'):
                    sizes = [int(x) for x in text.split()[1:]]
                elif text.startswith('TYPE'):
                    types = text.split()[1:]
                elif text.startswith('COUNT'):
                    counts = [int(x) for x in text.split()[1:]]
                elif text.startswith('POINTS'):
                    points = int(text.split()[1])
                elif text.startswith('DATA'):
                    data_type = text.split()[1]
                    break
            if data_type != 'binary':
                raise ValueError(f'Only binary PCD is supported: {filename}')
            dtype_fields = []
            for field, size, kind, count in zip(fields, sizes, types, counts):
                if kind == 'F' and size == 4:
                    dtype = np.float32
                elif kind == 'F' and size == 8:
                    dtype = np.float64
                elif kind == 'U' and size == 2:
                    dtype = np.uint16
                elif kind == 'U' and size == 4:
                    dtype = np.uint32
                elif kind == 'I' and size == 4:
                    dtype = np.int32
                else:
                    raise ValueError(f'Unsupported PCD field type {field}: {kind}{size}')
                dtype_fields.append((field, dtype, count) if count > 1 else (field, dtype))
            f.seek(header_len)
            pcd = np.fromfile(f, dtype=np.dtype(dtype_fields), count=points)
        return np.stack([pcd['x'], pcd['y'], pcd['z']], axis=1).astype(np.float32)

    def _read_lidar_xyz(self, sample):
        if self.lidar_source in {'raw', 'fake'}:
            return self._read_pcd_xyz(sample['lidar'])

        with np.load(sample['lidar']) as cloud:
            map_points = cloud['xyz'].astype(np.float64)
        position, map_from_body = self._interpolate_fastlio_pose(
            sample['calib']['trajectory'],
            sample['image_time_ns'],
        )
        body_points = (map_points - position) @ map_from_body
        body_h = np.concatenate(
            [body_points, np.ones((body_points.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        return (
            self.fastlio_body_to_lidar @ body_h.T
        ).T[:, :3].astype(np.float32)

    def _load_depth_m(self, filename):
        return np.array(Image.open(filename), dtype=np.float32) * self.depth_scale

    def _project_lidar_hint(self, sample, original_hw, target_hw, conversion_rate):
        if sample['lidar'] is None:
            return np.zeros(target_hw, dtype=np.float32), np.zeros(target_hw, dtype=bool)

        pts = self._read_lidar_xyz(sample)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_cam = (sample['calib']['lidar_to_cam'] @ np.concatenate([pts, ones], axis=1).T).T[:, :3]
        pts_rect = (sample['calib']['rect_r'] @ pts_cam.T).T
        pts_rect = pts_rect[pts_rect[:, 2] > 0.1]
        if pts_rect.size == 0:
            return np.zeros(target_hw, dtype=np.float32), np.zeros(target_hw, dtype=bool)

        k = sample['calib']['K']
        u = k[0, 0] * pts_rect[:, 0] / pts_rect[:, 2] + k[0, 2]
        v = k[1, 1] * pts_rect[:, 1] / pts_rect[:, 2] + k[1, 2]
        sx = target_hw[1] / float(original_hw[1])
        sy = target_hw[0] / float(original_hw[0])
        ui = np.rint(u * sx).astype(np.int32)
        vi = np.rint(v * sy).astype(np.int32)
        inside = (ui >= 0) & (ui < target_hw[1]) & (vi >= 0) & (vi < target_hw[0])
        ui, vi, z = ui[inside], vi[inside], pts_rect[:, 2][inside]
        if z.size == 0:
            return np.zeros(target_hw, dtype=np.float32), np.zeros(target_hw, dtype=bool)

        linear = vi * target_hw[1] + ui
        order = np.lexsort((z, linear))
        linear = linear[order]
        z = z[order]
        first = np.concatenate([[True], linear[1:] != linear[:-1]])
        linear = linear[first]
        z = z[first]

        depth_hint = np.zeros(target_hw[0] * target_hw[1], dtype=np.float32)
        depth_hint[linear] = z
        depth_hint = depth_hint.reshape(target_hw)
        disp_hint = np.zeros_like(depth_hint, dtype=np.float32)
        valid_hint = depth_hint > 0.1
        disp_hint[valid_hint] = conversion_rate / depth_hint[valid_hint]
        return disp_hint, valid_hint

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(1000)
                np.random.seed(1000)
                random.seed(1000)
                self.init_seed = True

        index = index % len(self.image_list)
        sample = self.extra_info[index]
        img1 = np.array(Image.open(sample['left']).convert('RGB')).astype(np.uint8)
        img2 = np.array(Image.open(sample['right']).convert('RGB')).astype(np.uint8)
        depth = self._load_depth_m(sample['depth'])
        original_hw = depth.shape
        target_hw = tuple(self.resize_hw) if self.resize_hw is not None else original_hw
        sx = target_hw[1] / float(original_hw[1])

        if target_hw != original_hw:
            img1 = np.array(Image.fromarray(img1).resize((target_hw[1], target_hw[0]), Image.BILINEAR))
            img2 = np.array(Image.fromarray(img2).resize((target_hw[1], target_hw[0]), Image.BILINEAR))
            depth = np.array(Image.fromarray(depth).resize((target_hw[1], target_hw[0]), Image.NEAREST), dtype=np.float32)

        conversion_rate = sample['calib']['conversion_rate'] * sx
        valid = depth > 0.01
        disp = np.zeros_like(depth, dtype=np.float32)
        disp[valid] = conversion_rate / depth[valid]
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if self.args.guided_flag:
            disp_hints, _ = self._project_lidar_hint(sample, original_hw, target_hw, conversion_rate)
            flow_hints = np.stack([disp_hints, np.zeros_like(disp_hints)], axis=-1)

        if self.border_crop_fraction > 0:
            height, width = flow.shape[:2]
            crop_y = int(round(height * self.border_crop_fraction))
            crop_x = int(round(width * self.border_crop_fraction))
            y_slice = slice(crop_y, height - crop_y)
            x_slice = slice(crop_x, width - crop_x)
            img1 = img1[y_slice, x_slice]
            img2 = img2[y_slice, x_slice]
            flow = flow[y_slice, x_slice]
            valid = valid[y_slice, x_slice]
            if self.args.guided_flag:
                flow_hints = flow_hints[y_slice, x_slice]

        if self.augmentor is not None:
            if self.args.guided_flag:
                img1, img2, flow, valid, flow_hints = self.augmentor(
                    img1, img2, flow, valid, flow_hints, more_bottom=self.args.more_bottom)
            else:
                img1, img2, flow, valid, flow_hints = self.augmentor(
                    img1, img2, flow, valid, more_bottom=self.args.more_bottom)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()[:1]
        valid = torch.from_numpy(valid)

        if self.img_pad is not None:
            pad_h, pad_w = self.img_pad
            img1 = F.pad(img1, [pad_w] * 2 + [pad_h] * 2)
            img2 = F.pad(img2, [pad_w] * 2 + [pad_h] * 2)

        conversion_rate = torch.tensor(conversion_rate).float()
        if self.args.guided_flag:
            flow_hints = torch.from_numpy(flow_hints).permute(2, 0, 1).float()[:1]
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float(), flow_hints, conversion_rate

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float(), conversion_rate


