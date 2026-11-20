import argparse
import glob
import json
import os
from collections import Counter
from scipy.optimize import linear_sum_assignment

import h5py
import numpy as np


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


def normalize_points_array(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 5:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 5), dtype=np.float32)
    return arr


def load_frames_and_flags(file_path):
    frames = []
    flags = []
    with h5py.File(file_path, "r") as f:
        points_ds = f["points_per_frame"]
        flags_ds = f["flags_per_frame"]
        num_frames = points_ds.shape[1]

        for frame_idx in range(num_frames):
            point_obj = read_cell_ref(f, points_ds, frame_idx)
            flag_obj = read_cell_ref(f, flags_ds, frame_idx)
            frames.append(normalize_points_array(point_obj))
            flags.append(decode_mat_char(flag_obj))
    return frames, flags


def axis_indices(mode):
    if mode == "yz":
        return (1, 2)
    if mode == "xz":
        return (0, 2)
    raise ValueError("mode must be 'yz' or 'xz'")


def pairwise_dist(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def dbscan_numpy(points_2d, eps=0.45, min_samples=3):
    n = len(points_2d)
    if n == 0:
        return np.array([], dtype=np.int32)

    distances = pairwise_dist(points_2d, points_2d)
    neighborhoods = [np.where(distances[i] <= eps)[0] for i in range(n)]

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for idx in range(n):
        if visited[idx]:
            continue
        visited[idx] = True
        neighbors = neighborhoods[idx]
        if len(neighbors) < min_samples:
            continue

        labels[idx] = cluster_id
        seeds = list(neighbors)
        seed_pos = 0

        while seed_pos < len(seeds):
            point_idx = seeds[seed_pos]
            seed_pos += 1
            if not visited[point_idx]:
                visited[point_idx] = True
                point_neighbors = neighborhoods[point_idx]
                if len(point_neighbors) >= min_samples:
                    for nb in point_neighbors:
                        if nb not in seeds:
                            seeds.append(nb)
            if labels[point_idx] == -1:
                labels[point_idx] = cluster_id

        cluster_id += 1

    return labels


def cluster_one_frame(frame_points, mode="yz", eps=0.45, min_samples=3, min_cluster_points=6):
    if frame_points.shape[0] == 0:
        return [], np.array([], dtype=np.int32)

    axes = axis_indices(mode)
    pts2d = frame_points[:, axes]
    labels = dbscan_numpy(pts2d, eps=eps, min_samples=min_samples)
    clusters = []

    for lb in np.unique(labels):
        if lb == -1:
            continue
        mask = labels == lb
        cluster_points = frame_points[mask]
        if cluster_points.shape[0] < min_cluster_points:
            labels[mask] = -1
            continue
        center_xyz = np.mean(cluster_points[:, :3], axis=0)
        clusters.append(
            {
                "cluster_id": int(lb),
                "center_xyz": center_xyz,
                "center_2d": center_xyz[list(axes)],
                "points": cluster_points,
                "num_points": int(cluster_points.shape[0]),
            }
        )

    return clusters, labels


def merge_close_clusters(clusters, mode="yz", merge_dist=0.8):
    if not clusters:
        return []

    merged = []
    used = [False] * len(clusters)
    axes = axis_indices(mode)

    for i in range(len(clusters)):
        if used[i]:
            continue

        points = [clusters[i]["points"]]
        used[i] = True
        center_i = clusters[i]["center_xyz"][list(axes)]

        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
            center_j = clusters[j]["center_xyz"][list(axes)]
            if np.linalg.norm(center_i - center_j) < merge_dist:
                points.append(clusters[j]["points"])
                used[j] = True

        merged_points = np.concatenate(points, axis=0)
        center_xyz = np.mean(merged_points[:, :3], axis=0)
        merged.append(
            {
                "cluster_id": len(merged),
                "center_xyz": center_xyz,
                "center_2d": center_xyz[list(axes)],
                "points": merged_points,
                "num_points": int(merged_points.shape[0]),
            }
        )

    return merged


def best_assignment(cost_matrix, max_cost):
    """Maximize gated matches, then minimize their total cost."""
    if cost_matrix.size == 0:
        return []
    num_tracks, num_dets = cost_matrix.shape
    valid = np.isfinite(cost_matrix) & (cost_matrix <= max_cost)
    if not valid.any():
        return []
    # One dummy column per track permits every track to remain unmatched.
    # This penalty makes one extra valid match preferable to any cost saving.
    span = float(np.max(np.abs(cost_matrix[valid]))) + 1.0
    penalty = (2 * min(num_tracks, num_dets) + 1) * span
    costs = np.full((num_tracks, num_dets + num_tracks), penalty, dtype=np.float64)
    costs[:, :num_dets] = np.where(valid, cost_matrix, 3 * penalty)
    rows, cols = linear_sum_assignment(costs)
    return [(int(t), int(d)) for t, d in zip(rows, cols) if d < num_dets and valid[t, d]]


class KalmanFilter2D:
    def __init__(self, init_x, init_y):
        self.x = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        dt = 1.0
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.02
        self.R = np.eye(2, dtype=np.float32) * 0.05

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, meas_x, meas_y):
        z = np.array([[meas_x], [meas_y]], dtype=np.float32)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(4, dtype=np.float32) - k @ self.H) @ self.P


class Track:
    def __init__(self, track_id, frame_idx, detection, frame_flag):
        self.track_id = track_id
        self.kf = KalmanFilter2D(detection["center_2d"][0], detection["center_2d"][1])
        self.last_frame = frame_idx
        self.missed = 0
        self.history = []
        self.avg_num_points = float(detection["num_points"])
        self.add_observation(frame_idx, detection, frame_flag)

    def predict(self):
        return self.kf.predict()

    def update(self, frame_idx, detection, frame_flag):
        self.kf.update(detection["center_2d"][0], detection["center_2d"][1])
        self.last_frame = frame_idx
        self.missed = 0
        self.avg_num_points = 0.8 * self.avg_num_points + 0.2 * float(detection["num_points"])
        self.add_observation(frame_idx, detection, frame_flag)

    def add_observation(self, frame_idx, detection, frame_flag):
        self.history.append(
            {
                "frame_idx": int(frame_idx),
                "flag": frame_flag,
                "center_xyz": detection["center_xyz"].astype(np.float32),
                "center_2d": detection["center_2d"].astype(np.float32),
                "points": detection["points"].astype(np.float32),
                "num_points": int(detection["num_points"]),
            }
        )

    def mark_missed(self):
        self.missed += 1


class MultiObjectTracker:
    def __init__(self, dist_threshold=0.9, max_missed=10, max_points_ratio=2.8):
        self.dist_threshold = dist_threshold
        self.max_missed = max_missed
        self.max_points_ratio = max_points_ratio
        self.tracks = []
        self.finished_tracks = []
        self.next_id = 0

    def update(self, detections, frame_idx, frame_flag):
        predictions = [track.predict() for track in self.tracks]

        if self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, pred in enumerate(predictions):
                for d_idx, det in enumerate(detections):
                    center_dist = np.linalg.norm(pred - det["center_2d"])
                    point_ratio = max(
                        det["num_points"] / max(self.tracks[t_idx].avg_num_points, 1.0),
                        self.tracks[t_idx].avg_num_points / max(det["num_points"], 1.0),
                    )
                    if point_ratio > self.max_points_ratio:
                        cost_matrix[t_idx, d_idx] = self.dist_threshold + 1.0
                    else:
                        point_penalty = 0.08 * abs(np.log(max(point_ratio, 1e-6)))
                        cost_matrix[t_idx, d_idx] = center_dist + point_penalty
            pairs = best_assignment(cost_matrix, self.dist_threshold)
        else:
            pairs = []

        assigned_tracks = set()
        assigned_dets = set()
        for track_idx, det_idx in pairs:
            self.tracks[track_idx].update(frame_idx, detections[det_idx], frame_flag)
            assigned_tracks.add(track_idx)
            assigned_dets.add(det_idx)

        for track_idx, track in enumerate(self.tracks):
            if track_idx not in assigned_tracks:
                track.mark_missed()

        for det_idx, det in enumerate(detections):
            if det_idx not in assigned_dets:
                self.tracks.append(Track(self.next_id, frame_idx, det, frame_flag))
                self.next_id += 1

        still_active = []
        for track in self.tracks:
            if track.missed > self.max_missed:
                self.finished_tracks.append(track)
            else:
                still_active.append(track)
        self.tracks = still_active

    def all_tracks(self):
        return self.finished_tracks + self.tracks


def split_track_on_jumps(track, max_step_dist=1.1, min_segment_len=8):
    if len(track.history) < min_segment_len:
        return []
    if len(track.history) == 1:
        return [track]

    split_points = [0]
    for idx in range(1, len(track.history)):
        prev_center = track.history[idx - 1]["center_2d"]
        curr_center = track.history[idx]["center_2d"]
        step_dist = float(np.linalg.norm(curr_center - prev_center))
        frame_gap = int(track.history[idx]["frame_idx"]) - int(track.history[idx - 1]["frame_idx"])
        if step_dist > max_step_dist or frame_gap > 3:
            split_points.append(idx)
    split_points.append(len(track.history))

    segments = []
    for seg_idx in range(len(split_points) - 1):
        start = split_points[seg_idx]
        end = split_points[seg_idx + 1]
        segment_history = track.history[start:end]
        if len(segment_history) < min_segment_len:
            continue
        new_track = Track.__new__(Track)
        new_track.track_id = track.track_id
        new_track.source_track_id = track.track_id
        new_track.kf = None
        new_track.last_frame = segment_history[-1]["frame_idx"]
        new_track.missed = 0
        new_track.history = segment_history
        new_track.avg_num_points = float(np.mean([item["num_points"] for item in segment_history]))
        segments.append(new_track)
    return segments


def refine_tracks(tracks, max_step_dist=1.1, min_track_len=8):
    refined = []
    for track in tracks:
        refined.extend(split_track_on_jumps(track, max_step_dist=max_step_dist, min_segment_len=min_track_len))
    refined.sort(key=lambda item: len(item.history), reverse=True)
    for track_id, track in enumerate(refined):
        track.track_id = track_id
    return refined


def frame_limits(mode):
    if mode == "yz":
        return (-5, 5), (-3, 3), "y", "z"
    return (-4, 4), (-1, 3), "x", "z"


def color_for_index(index):
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
    ]
    return palette[index % len(palette)]


def map_to_canvas(x, y, xlim, ylim, width, height, padding):
    draw_w = width - 2 * padding
    draw_h = height - 2 * padding
    px = padding + (x - xlim[0]) / (xlim[1] - xlim[0]) * draw_w
    py = height - padding - (y - ylim[0]) / (ylim[1] - ylim[0]) * draw_h
    return px, py


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def svg_footer(parts):
    parts.append("</svg>")
    return "\n".join(parts)


def draw_axes(parts, xlim, ylim, xlabel, ylabel, width, height, padding):
    x0, y0 = map_to_canvas(xlim[0], ylim[0], xlim, ylim, width, height, padding)
    x1, _ = map_to_canvas(xlim[1], ylim[0], xlim, ylim, width, height, padding)
    _, y1 = map_to_canvas(xlim[0], ylim[1], xlim, ylim, width, height, padding)
    parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" stroke="black" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="black" stroke-width="1.2"/>')
    parts.append(f'<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="14">{xlabel}</text>')
    parts.append(f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {height / 2:.1f})">{ylabel}</text>')


def add_svg_points(parts, pts2d, color, xlim, ylim, width, height, padding, radius=2.4, opacity=0.9):
    for x, y in pts2d:
        px, py = map_to_canvas(float(x), float(y), xlim, ylim, width, height, padding)
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{color}" fill-opacity="{opacity}"/>')


def add_svg_rect(parts, pts2d, color, xlim, ylim, width, height, padding, label=None):
    xmin, ymin = np.min(pts2d, axis=0)
    xmax, ymax = np.max(pts2d, axis=0)
    x1, y1 = map_to_canvas(float(xmin), float(ymax), xlim, ylim, width, height, padding)
    x2, y2 = map_to_canvas(float(xmax), float(ymin), xlim, ylim, width, height, padding)
    rect_x = min(x1, x2)
    rect_y = min(y1, y2)
    rect_w = max(abs(x2 - x1), 4)
    rect_h = max(abs(y2 - y1), 4)
    parts.append(f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" width="{rect_w:.2f}" height="{rect_h:.2f}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    if label is not None:
        parts.append(f'<text x="{rect_x:.2f}" y="{max(rect_y - 4, 14):.2f}" font-size="11" fill="{color}">{label}</text>')


def add_svg_polyline(parts, pts2d, color, xlim, ylim, width, height, padding, label=None):
    if len(pts2d) == 0:
        return
    svg_points = []
    for x, y in pts2d:
        px, py = map_to_canvas(float(x), float(y), xlim, ylim, width, height, padding)
        svg_points.append(f"{px:.2f},{py:.2f}")
    parts.append(f'<polyline points="{" ".join(svg_points)}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    if label is not None:
        px, py = svg_points[0].split(",")
        parts.append(f'<text x="{px}" y="{float(py) - 4:.2f}" font-size="11" fill="{color}">{label}</text>')


def save_frame_plot(output_path, frame_points, labels, tracks, frame_idx, frame_flag, mode="yz"):
    width, height, padding = 720, 620, 60
    xlim, ylim, xlabel, ylabel = frame_limits(mode)
    axes = axis_indices(mode)
    parts = svg_header(width, height)
    draw_axes(parts, xlim, ylim, xlabel, ylabel, width, height, padding)
    parts.append(f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="18">frame {frame_idx} | flag {frame_flag}</text>')

    if frame_points.shape[0] > 0:
        pts2d = frame_points[:, axes]
        unique_labels = np.unique(labels) if len(labels) else np.array([-1])
        for color_idx, lb in enumerate(unique_labels):
            mask = labels == lb
            pts = pts2d[mask]
            if pts.size == 0:
                continue
            if lb == -1:
                add_svg_points(parts, pts, "#c0c0c0", xlim, ylim, width, height, padding, radius=1.8, opacity=0.8)
            else:
                add_svg_points(parts, pts, color_for_index(color_idx), xlim, ylim, width, height, padding)

    for track in tracks:
        if track.history and track.history[-1]["frame_idx"] == frame_idx:
            obs = track.history[-1]
            pts2d = obs["points"][:, axes]
            add_svg_rect(parts, pts2d, "#000000", xlim, ylim, width, height, padding, label=f"ID {track.track_id}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_footer(parts))


def save_track_paths(output_path, tracks, mode="yz"):
    width, height, padding = 720, 620, 60
    xlim, ylim, xlabel, ylabel = frame_limits(mode)
    parts = svg_header(width, height)
    draw_axes(parts, xlim, ylim, xlabel, ylabel, width, height, padding)
    parts.append(f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="18">Track center paths</text>')

    for idx, track in enumerate(tracks):
        centers = np.array([item["center_2d"] for item in track.history], dtype=np.float32)
        if len(centers) == 0:
            continue
        color = color_for_index(idx)
        add_svg_polyline(parts, centers, color, xlim, ylim, width, height, padding, label=f"ID {track.track_id}")
        add_svg_points(parts, centers, color, xlim, ylim, width, height, padding, radius=2.0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_footer(parts))


def build_track_summary(tracks):
    summary = []
    for track in tracks:
        flags = [item["flag"] for item in track.history]
        pos_frames = sum(flag.endswith("-1") for flag in flags)
        neg_frames = sum(flag.endswith("-0") for flag in flags)
        centers = np.array([item["center_2d"] for item in track.history], dtype=np.float32)
        path_length = 0.0
        if len(centers) > 1:
            path_length = float(np.sum(np.linalg.norm(centers[1:] - centers[:-1], axis=1)))
        summary.append(
            {
                "track_id": int(track.track_id),
                "source_track_id": int(getattr(track, "source_track_id", track.track_id)),
                "track_len": int(len(track.history)),
                "start_frame": int(track.history[0]["frame_idx"]),
                "end_frame": int(track.history[-1]["frame_idx"]),
                "positive_frames": int(pos_frames),
                "negative_frames": int(neg_frames),
                "positive_ratio": float(pos_frames / max(len(track.history), 1)),
                "path_length_2d": path_length,
                "flag_hist": dict(Counter(flags)),
            }
        )
    summary.sort(key=lambda item: item["track_len"], reverse=True)
    return summary


def export_tracks_npz(output_path, source_file, tracks):
    track_list = []
    for track in tracks:
        track_list.append(
            {
                "track_id": int(track.track_id),
                "history": track.history,
            }
        )
    np.savez(output_path, track_list=np.array(track_list, dtype=object), source_file=source_file)


def process_file(file_path, output_root, args):
    frames, flags = load_frames_and_flags(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    case_dir = os.path.join(output_root, base_name)
    frames_dir = os.path.join(case_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    tracker = MultiObjectTracker(
        dist_threshold=args.dist_threshold,
        max_missed=args.max_missed,
        max_points_ratio=args.max_points_ratio,
    )
    cluster_counts = []

    for frame_idx, (frame_points, frame_flag) in enumerate(zip(frames, flags)):
        clusters, labels = cluster_one_frame(
            frame_points,
            mode=args.mode,
            eps=args.eps,
            min_samples=args.min_samples,
            min_cluster_points=args.min_cluster_points,
        )
        clusters = merge_close_clusters(clusters, mode=args.mode, merge_dist=args.merge_dist)
        tracker.update(clusters, frame_idx, frame_flag)
        cluster_counts.append(len(clusters))

        if args.save_every > 0 and frame_idx % args.save_every == 0:
            output_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.svg")
            save_frame_plot(output_path, frame_points, labels, tracker.all_tracks(), frame_idx, frame_flag, mode=args.mode)

    tracks = [track for track in tracker.all_tracks() if len(track.history) >= args.min_track_len]
    tracks = refine_tracks(tracks, max_step_dist=args.max_step_dist, min_track_len=args.min_track_len)
    summary = build_track_summary(tracks)

    save_track_paths(os.path.join(case_dir, "track_paths.svg"), tracks, mode=args.mode)
    export_tracks_npz(os.path.join(case_dir, "rebuilt_tracks.npz"), file_path, tracks)

    run_summary = {
        "source_file": file_path,
        "num_frames": len(frames),
        "flag_hist": dict(Counter(flags)),
        "cluster_count_hist": dict(Counter(cluster_counts)),
        "num_tracks_after_filter": len(tracks),
        "tracks": summary,
        "params": {
            "mode": args.mode,
            "eps": args.eps,
            "min_samples": args.min_samples,
            "min_cluster_points": args.min_cluster_points,
            "merge_dist": args.merge_dist,
            "dist_threshold": args.dist_threshold,
            "max_missed": args.max_missed,
            "max_points_ratio": args.max_points_ratio,
            "max_step_dist": args.max_step_dist,
            "min_track_len": args.min_track_len,
        },
    }

    with open(os.path.join(case_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(case_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"source_file: {file_path}\n")
        f.write(f"num_frames: {len(frames)}\n")
        f.write(f"flag_hist: {dict(Counter(flags))}\n")
        f.write(f"cluster_count_hist: {dict(Counter(cluster_counts))}\n")
        f.write(f"num_tracks_after_filter: {len(tracks)}\n\n")
        for item in summary:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return run_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone multi-person track debugger")
    parser.add_argument(
        "--input",
        default=os.path.join(POINTNET_ROOT, "processed_data"),
        help="Absolute input folder containing *_frames.mat files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(POINTNET_ROOT, "debug_tools", "outputs"),
        help="Absolute output folder for debug artifacts",
    )
    parser.add_argument("--pattern", default="*_frames.mat", help="Glob pattern inside input folder")
    parser.add_argument("--keyword", default="", help="Only process files containing this keyword")
    parser.add_argument("--mode", choices=["yz", "xz"], default="yz")
    parser.add_argument("--eps", type=float, default=0.45)
    parser.add_argument("--min-samples", dest="min_samples", type=int, default=3)
    parser.add_argument("--min-cluster-points", dest="min_cluster_points", type=int, default=6)
    parser.add_argument("--merge-dist", dest="merge_dist", type=float, default=0.8)
    parser.add_argument("--dist-threshold", dest="dist_threshold", type=float, default=0.9)
    parser.add_argument("--max-missed", dest="max_missed", type=int, default=10)
    parser.add_argument("--max-points-ratio", dest="max_points_ratio", type=float, default=2.8)
    parser.add_argument("--max-step-dist", dest="max_step_dist", type=float, default=1.1)
    parser.add_argument("--min-track-len", dest="min_track_len", type=int, default=8)
    parser.add_argument("--save-every", dest="save_every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    input_pattern = os.path.join(args.input, args.pattern)
    files = sorted(glob.glob(input_pattern))
    if args.keyword:
        files = [file_path for file_path in files if args.keyword in os.path.basename(file_path)]
    if not files:
        raise RuntimeError(f"No files found for pattern: {input_pattern}")

    os.makedirs(args.output, exist_ok=True)
    all_runs = []
    for file_path in files:
        print(f"processing: {file_path}")
        all_runs.append(process_file(file_path, args.output, args))

    with open(os.path.join(args.output, "run_index.json"), "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)

    print(f"saved outputs to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
