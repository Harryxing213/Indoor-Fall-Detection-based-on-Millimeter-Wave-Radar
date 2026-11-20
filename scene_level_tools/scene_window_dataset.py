import glob
import os

import h5py
import numpy as np
from torch.utils.data import Dataset
from data_utils.file_filters import is_single_person, is_multi_person


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


def decode_mat_char(ds):
    arr = np.array(ds)
    if arr.dtype.kind in ("u", "i"):
        arr = arr.astype(np.int64).flatten()
        return "".join(chr(x) for x in arr if x != 0)
    return str(arr)


def read_cell_ref(file_handle, cell_dataset, idx):
    ref = cell_dataset[0, idx]
    return file_handle[ref]


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if scale > 0:
        pc = pc / scale
    return pc.astype(np.float32)


def farthest_point_sample(points, npoint):
    n, _ = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((n,), dtype=np.float32) * 1e10
    farthest = np.random.randint(0, n)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance, axis=-1))

    return points[centroids]


class SceneWindowDataset(Dataset):
    def __init__(
        self,
        root,
        args,
        split="train",
        test_ratio=0.25,
        window_size=12,
        stride=4,
        min_segment_len=6,
        include_single=False,
        include_multi_only=True,
    ):
        super().__init__()
        self.root = root
        self.npoints = args.num_point
        self.window_size = window_size
        self.stride = stride
        self.min_segment_len = min_segment_len

        files = sorted(glob.glob(os.path.join(root, "*_frames.mat")))
        if include_multi_only:
            files = [f for f in files if is_multi_person(f)]
        elif not include_single:
            files = [f for f in files if not is_single_person(f)]
        if not include_single:
            files = [f for f in files if not is_single_person(f)]
        if not files:
            raise RuntimeError(f"No valid *_frames.mat files found in {root}")

        split_idx = int(len(files) * (1 - test_ratio))
        selected_files = files[:split_idx] if split == "train" else files[split_idx:]
        if not selected_files:
            raise RuntimeError(f"No files selected for split={split}")

        self.samples = []
        self.labels = []
        for file_path in selected_files:
            with h5py.File(file_path, "r") as f:
                flags_ds = f["flags_per_frame"]
                num_frames = flags_ds.shape[1]
                flags = []
                for frame_idx in range(num_frames):
                    flag_obj = read_cell_ref(f, flags_ds, frame_idx)
                    flags.append(decode_mat_char(flag_obj))

            for segment in self._build_segments(flags):
                seg_len = segment["end"] - segment["start"] + 1
                if seg_len < max(window_size, min_segment_len):
                    continue
                label = 1 if segment["flag"].endswith("-1") else 0
                for start_frame in range(segment["start"], segment["end"] - window_size + 2, stride):
                    end_frame = start_frame + window_size - 1
                    self.samples.append(
                        {
                            "file_path": file_path,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "label": label,
                            "flag": segment["flag"],
                        }
                    )
                    self.labels.append(label)

        if not self.samples:
            raise RuntimeError("No valid scene windows found")

        self.labels = np.array(self.labels, dtype=np.int64)
        unique, counts = np.unique(self.labels, return_counts=True)
        label_hist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"[{split}] scene windows={len(self.samples)} label_hist={label_hist}")

    def _build_segments(self, flags):
        segments = []
        if not flags:
            return segments
        current_flag = flags[0]
        seg_start = 0
        for idx in range(1, len(flags)):
            if flags[idx] != current_flag:
                segments.append({"start": seg_start, "end": idx - 1, "flag": current_flag})
                seg_start = idx
                current_flag = flags[idx]
        segments.append({"start": seg_start, "end": len(flags) - 1, "flag": current_flag})
        return segments

    def __len__(self):
        return len(self.samples)

    def _sample_points(self, pts_xyz):
        num_points = pts_xyz.shape[0]
        if num_points == 0:
            return np.zeros((self.npoints, 3), dtype=np.float32)
        if num_points >= self.npoints:
            return farthest_point_sample(pts_xyz, self.npoints).astype(np.float32)
        repeat_idx = np.random.choice(num_points, self.npoints - num_points, replace=True)
        padded = np.concatenate([pts_xyz, pts_xyz[repeat_idx, :]], axis=0)
        shuffle_idx = np.random.permutation(self.npoints)
        return padded[shuffle_idx].astype(np.float32)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        merged_points = []
        with h5py.File(meta["file_path"], "r") as f:
            points_ds = f["points_per_frame"]
            for frame_idx in range(meta["start_frame"], meta["end_frame"] + 1):
                point_obj = read_cell_ref(f, points_ds, frame_idx)
                arr = np.array(point_obj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == 5:
                    arr = arr.T
                if arr.ndim != 2 or arr.shape[1] < 3 or arr.size == 0:
                    continue
                merged_points.append(arr[:, :3])

        if not merged_points:
            pts_xyz = np.zeros((self.npoints, 3), dtype=np.float32)
        else:
            pts_xyz = np.concatenate(merged_points, axis=0)
            pts_xyz = self._sample_points(pts_xyz)
            pts_xyz = pc_normalize(pts_xyz)

        return pts_xyz.astype(np.float32), np.int64(meta["label"]), meta
