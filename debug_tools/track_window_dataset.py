import glob
import os
from collections import Counter

import numpy as np
from torch.utils.data import Dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


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


class MultiPersonTrackWindowDataset(Dataset):
    def __init__(
        self,
        root,
        args,
        split="train",
        test_ratio=0.25,
        window_size=15,
        stride=5,
        min_track_len=10,
        include_single=False,
        include_multi_only=True,
        min_points_per_window=24,
        require_monotonic_tracks=True,
    ):
        super().__init__()
        self.root = root
        self.npoints = args.num_point
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.min_track_len = min_track_len
        self.min_points_per_window = min_points_per_window
        self.require_monotonic_tracks = require_monotonic_tracks

        self.case_files = self._discover_case_files(root)
        self.case_files = self._filter_case_files(self.case_files, include_single, include_multi_only)
        if not self.case_files:
            raise RuntimeError("No rebuilt_tracks.npz files found after filtering")

        split_idx = int(len(self.case_files) * (1 - test_ratio))
        if split == "train":
            selected_files = self.case_files[:split_idx]
        else:
            selected_files = self.case_files[split_idx:]

        if not selected_files:
            raise RuntimeError(f"No files selected for split={split}")

        self.samples = []
        label_counter = Counter()
        skipped_ambiguous = 0
        skipped_non_monotonic = 0

        for file_path in selected_files:
            data = np.load(file_path, allow_pickle=True)
            track_list = data["track_list"]
            source_file = str(data["source_file"])

            for tr in track_list:
                tr = tr.item() if isinstance(tr, np.ndarray) else tr
                history = sorted(tr["history"], key=lambda item: item["frame_idx"])
                if len(history) < max(self.window_size, self.min_track_len):
                    continue

                binary_flags = self._flags_to_binary(history)
                if self.require_monotonic_tracks and not self._is_monotonic_track(binary_flags):
                    skipped_non_monotonic += 1
                    continue

                segments = self._build_binary_segments(history, binary_flags)
                for segment in segments:
                    seg_history = segment["history"]
                    seg_label = segment["label"]
                    if len(seg_history) < self.window_size:
                        continue

                    for start in range(0, len(seg_history) - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        history_window = seg_history[start:end]
                        label = self._infer_window_label(history_window)
                        if label is None:
                            skipped_ambiguous += 1
                            continue
                        if int(label) != int(seg_label):
                            skipped_ambiguous += 1
                            continue
                        self.samples.append(
                            {
                                "file_path": file_path,
                                "source_file": source_file,
                                "track_id": int(tr["track_id"]),
                                "start_frame": int(history_window[0]["frame_idx"]),
                                "end_frame": int(history_window[-1]["frame_idx"]),
                                "label": int(label),
                            }
                        )
                        label_counter[int(label)] += 1

        if not self.samples:
            raise RuntimeError("No valid track windows built. Try relaxing thresholds.")

        self.labels = np.array([sample["label"] for sample in self.samples], dtype=np.int64)

        print(
            f"[{split}] files={len(selected_files)} windows={len(self.samples)} "
            f"label_hist={dict(label_counter)} skipped_ambiguous={skipped_ambiguous} "
            f"skipped_non_monotonic={skipped_non_monotonic}"
        )

    def _discover_case_files(self, root):
        direct = sorted(glob.glob(os.path.join(root, "**", "rebuilt_tracks.npz"), recursive=True))
        if direct:
            return direct
        fallback = sorted(glob.glob(os.path.join(root, "*_tracks.npz")))
        return fallback

    def _filter_case_files(self, file_paths, include_single, include_multi_only):
        filtered = []
        for path in file_paths:
            name = os.path.basename(path)
            source_hint = os.path.basename(os.path.dirname(path)) + name
            is_multi = ("双人" in source_hint) or ("三人" in source_hint)
            is_single = "一人" in source_hint and not is_multi
            if include_multi_only and not is_multi:
                continue
            if not include_single and is_single:
                continue
            filtered.append(path)
        return filtered

    def _infer_window_label(self, history_window):
        binary_flags = self._flags_to_binary(history_window)
        unique_flags = set(binary_flags)
        if len(unique_flags) != 1:
            return None
        return int(binary_flags[0])

    def _flags_to_binary(self, history_window):
        return [1 if str(item.get("flag", "")).endswith("-1") else 0 for item in history_window]

    def _is_monotonic_track(self, binary_flags):
        if not binary_flags:
            return False
        for prev_flag, curr_flag in zip(binary_flags[:-1], binary_flags[1:]):
            if prev_flag == 1 and curr_flag == 0:
                return False
        return True

    def _build_binary_segments(self, history, binary_flags):
        segments = []
        if not history:
            return segments
        start = 0
        curr = binary_flags[0]
        for idx in range(1, len(history)):
            if binary_flags[idx] != curr:
                segments.append({"label": int(curr), "history": history[start:idx]})
                start = idx
                curr = binary_flags[idx]
        segments.append({"label": int(curr), "history": history[start:]})
        return segments

    def __len__(self):
        return len(self.samples)

    def _sample_points(self, pts_xyz):
        k = pts_xyz.shape[0]
        if k == 0:
            return np.zeros((self.npoints, 3), dtype=np.float32)
        if k >= self.npoints:
            return farthest_point_sample(pts_xyz, self.npoints).astype(np.float32)
        repeat_idx = np.random.choice(k, self.npoints - k, replace=True)
        padded = np.concatenate([pts_xyz, pts_xyz[repeat_idx, :]], axis=0)
        shuffle_idx = np.random.permutation(self.npoints)
        return padded[shuffle_idx].astype(np.float32)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        data = np.load(meta["file_path"], allow_pickle=True)
        track_list = data["track_list"]

        target_track = None
        for tr in track_list:
            tr = tr.item() if isinstance(tr, np.ndarray) else tr
            if int(tr["track_id"]) == meta["track_id"]:
                target_track = tr
                break

        if target_track is None:
            raise RuntimeError(f"track_id {meta['track_id']} not found in {meta['file_path']}")

        history = sorted(target_track["history"], key=lambda item: item["frame_idx"])
        history_window = [
            item
            for item in history
            if meta["start_frame"] <= int(item["frame_idx"]) <= meta["end_frame"]
        ]

        merged_points = []
        total_points = 0
        for item in history_window:
            pts = np.array(item["points"], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                continue
            xyz = pts[:, :3]
            merged_points.append(xyz)
            total_points += xyz.shape[0]

        if total_points < self.min_points_per_window or not merged_points:
            pts_xyz = np.zeros((self.npoints, 3), dtype=np.float32)
        else:
            pts_xyz = np.concatenate(merged_points, axis=0)
            pts_xyz = self._sample_points(pts_xyz)
            pts_xyz = pc_normalize(pts_xyz)

        return (
            pts_xyz.astype(np.float32),
            np.int64(meta["label"]),
            meta,
        )
