import os
import numpy as np
import glob
import h5py

from tqdm import tqdm
from torch.utils.data import Dataset
from data_utils.file_filters import is_single_person


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def decode_mat_char(ds):
    arr = np.array(ds)
    if arr.dtype.kind in ("u","i"):
        arr = arr.astype(np.int64).flatten()
        return "".join(chr(x) for x in arr if x != 0)
    return str(arr)


def read_cell_ref(file_handle, cell_dataset, idx):
    """
    cell_dataset shape usually [1, T]
    return the referenced object content
    """
    ref = cell_dataset[0, idx]
    return file_handle[ref]


class RadarFallDataLoader(Dataset):
    """
    Read processed *_frames.mat files produced by MATLAB.

    Each saved file contains:
      - points_per_frame{t}: [num_points_t, 5] in MATLAB
        but h5py usually reads it back as [5, num_points_t], so we transpose.
      - flags_per_frame{t}: 'm-n'
      - optionally segments (we don't rely on them; we rebuild from flags)

    Returned sample:
      points: [num_point, 3]
      label: int
    """

    def __init__(self, root, args, split = "train", task = "fall_binary", test_ratio = 0.2,
                 window_size = 15, stride = 5, min_segment_len = 5, single_only = False, legacy_name_filter = False):
        super().__init__()
        self.root = root
        self.args = args
        self.npoints = args.num_point
        self.split = split
        self.task = task
        self.window_size = window_size
        self.stride = stride
        self.min_segment_len = min_segment_len
        self.single_only = single_only

        if task == "fall_binary":
            self.label_map = {
                "1-0":0,
                "2-0":0,
                "3-0":0,
                "1-1":1,
                "2-1":1,
                "3-1":1,
            }
        else:
            raise ValueError("Unsupported task: {}".format(task))

        if window_size < 1 or stride < 1 or not 0 < test_ratio < 1:
            raise ValueError("Use positive window/stride and 0 < test_ratio < 1")
        self.files = sorted(glob.glob(os.path.join(root, "*_frames.mat")))
        if self.single_only:
            self.files = [f for f in self.files if ('一人' in os.path.basename(f) if legacy_name_filter else is_single_person(f))]
        if len(self.files) == 0:
            raise RuntimeError("No files found in {}".format(root))
        split_idx = int(len(self.files) * (1-test_ratio))
        if split == "train":
            selected_files = self.files[:split_idx]
        else:
            selected_files = self.files[split_idx:]

        self.selected_files = selected_files
        self.samples = []
        all_window = []

        for file_path in selected_files:
            with h5py.File(file_path, "r") as f:
                flags_ds = f["flags_per_frame"]
                num_frames = flags_ds.shape[1]

                flags = []
                for t in range(num_frames):
                    flag_obj = read_cell_ref(f, flags_ds, t)
                    flag_str = decode_mat_char(flag_obj)
                    flags.append(flag_str)

                segments = self._build_segments(flags)
                for seg in segments:
                    seg_flag = seg["flag"]
                    if seg_flag not in self.label_map:
                        continue
                    seg_len = seg["end"] - seg["start"] + 1
                    if seg_len < max(self.window_size, self.min_segment_len):
                        continue
                    start_frame = seg["start"]
                    end_frame = seg["end"]

                    for ws in range(start_frame,end_frame-self.window_size+2, self.stride):
                        we = ws + self.window_size - 1
                        all_window.append(
                            {
                                "file_path":file_path,
                                "start_frame":ws,
                                "end_frame":we,
                                "flag":seg_flag,
                                "label":self.label_map[seg_flag],
                            }
                        )
        if len(all_window) == 0:
            raise RuntimeError("No valid windows found in the processed data")
        self.samples = all_window
        print(f"[{split}] task={task}, total windows={len(self.samples)}")

    def _build_segments(self, flags):
        segments = []
        if len(flags) == 0:
            return segments

        curr_flag = flags[0]
        seg_start = 0

        for i in range(1, len(flags)):
            if flags[i] != curr_flag:
                segments.append(
                    {
                        "start": seg_start,
                        "end": i - 1,
                        "flag": curr_flag,
                    }
                )
                seg_start = i
                curr_flag = flags[i]

        segments.append(
            {
                "start": seg_start,
                "end": len(flags) - 1,
                "flag": curr_flag,
            }
        )
        return segments

    def __len__(self):
        return len(self.samples)

    def _sample_points(self, pts_xyz):
        """
        pts_xyz: [K, 3]
        return: [num_point, 3]
        """
        k = pts_xyz.shape[0]
        if k == 0:
            return np.zeros((self.npoints, 3), dtype=np.float32)

        if k >= self.npoints:
            pts_xyz = farthest_point_sample(pts_xyz, self.npoints)
            return pts_xyz.astype(np.float32)
        else:
            repeat_idx = np.random.choice(k, self.npoints - k, replace=True)
            pad_points = pts_xyz[repeat_idx, :]
            pts_xyz = np.concatenate([pts_xyz, pad_points], axis=0)
            return pts_xyz.astype(np.float32)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        file_path = meta["file_path"]
        start_frame = meta["start_frame"]
        end_frame = meta["end_frame"]
        label = meta["label"]

        merged_points = []

        with h5py.File(file_path, "r") as f:
            points_ds = f["points_per_frame"]

            for t in range(start_frame, end_frame + 1):
                point_obj = read_cell_ref(f, points_ds, t)
                arr = np.array(point_obj, dtype=np.float32)

                # h5py usually sees MATLAB [M,5] as [5,M], so transpose
                if arr.ndim == 2 and arr.shape[0] == 5:
                    arr = arr.T

                # now arr should be [num_points_t, 5]
                if arr.size == 0:
                    continue

                if arr.ndim != 2 or arr.shape[1] < 3:
                    continue

                xyz = arr[:, :3]
                merged_points.append(xyz)

        if len(merged_points) == 0:
            pts_xyz = np.zeros((self.npoints, 3), dtype=np.float32)
        else:
            pts_xyz = np.concatenate(merged_points, axis=0)
            pts_xyz = self._sample_points(pts_xyz)
            pts_xyz = pc_normalize(pts_xyz)

        return pts_xyz.astype(np.float32), np.int64(label)
