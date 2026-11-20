import argparse
import importlib
import json
import os
import sys
from collections import Counter

import numpy as np
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEBUG_TOOLS_DIR = os.path.join(POINTNET_ROOT, "debug_tools")
MODELS_DIR = os.path.join(POINTNET_ROOT, "models")

sys.path.append(POINTNET_ROOT)
sys.path.append(DEBUG_TOOLS_DIR)
sys.path.append(MODELS_DIR)

from track_debugger import (  # noqa: E402
    MultiObjectTracker,
    cluster_one_frame,
    load_frames_and_flags,
    merge_close_clusters,
    refine_tracks,
)


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


def sample_window_points(history_window, num_point):
    merged_points = []
    for item in history_window:
        pts = np.array(item["points"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
            continue
        merged_points.append(pts[:, :3])

    if not merged_points:
        return np.zeros((num_point, 3), dtype=np.float32)

    pts_xyz = np.concatenate(merged_points, axis=0)
    if pts_xyz.shape[0] >= num_point:
        pts_xyz = farthest_point_sample(pts_xyz, num_point)
    else:
        repeat_idx = np.random.choice(pts_xyz.shape[0], num_point - pts_xyz.shape[0], replace=True)
        pts_xyz = np.concatenate([pts_xyz, pts_xyz[repeat_idx, :]], axis=0)
    return pc_normalize(pts_xyz)


def build_track_windows(track, window_size, stride, num_point):
    history = sorted(track.history, key=lambda item: item["frame_idx"])
    windows = []
    if len(history) < window_size:
        return windows
    for start in range(0, len(history) - window_size + 1, stride):
        end = start + window_size
        history_window = history[start:end]
        points = sample_window_points(history_window, num_point)
        windows.append(
            {
                "track_id": int(track.track_id),
                "start_frame": int(history_window[0]["frame_idx"]),
                "end_frame": int(history_window[-1]["frame_idx"]),
                "points": points,
                "flags": [item["flag"] for item in history_window],
            }
        )
    return windows


def load_classifier(model_name, checkpoint_path, num_category, use_normals, device):
    model_lib = importlib.import_module(model_name)
    classifier = model_lib.get_model(num_category, normal_channel=use_normals)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    classifier.eval()
    return classifier


def infer_windows(classifier, windows, device):
    results = []
    with torch.no_grad():
        for item in windows:
            points = torch.tensor(item["points"], dtype=torch.float32).unsqueeze(0).transpose(2, 1).to(device)
            pred, _ = classifier(points)
            probs = torch.exp(pred).cpu().numpy()[0]
            pred_label = int(np.argmax(probs))
            results.append(
                {
                    "track_id": item["track_id"],
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                    "pred_label": pred_label,
                    "fall_prob": float(probs[1]),
                    "non_fall_prob": float(probs[0]),
                    "flags": item["flags"],
                }
            )
    return results


def summarize_track_predictions(predictions, threshold):
    grouped = {}
    for item in predictions:
        grouped.setdefault(item["track_id"], []).append(item)

    summary = []
    for track_id, items in grouped.items():
        fall_probs = [item["fall_prob"] for item in items]
        pred_labels = [item["pred_label"] for item in items]
        summary.append(
            {
                "track_id": int(track_id),
                "num_windows": len(items),
                "mean_fall_prob": float(np.mean(fall_probs)),
                "max_fall_prob": float(np.max(fall_probs)),
                "positive_windows": int(sum(label == 1 for label in pred_labels)),
                "final_decision": int(np.max(fall_probs) >= threshold),
            }
        )
    summary.sort(key=lambda item: item["max_fall_prob"], reverse=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-person classifier on multiperson tracks")
    parser.add_argument("--input", default=os.path.join(POINTNET_ROOT, "processed_data"))
    parser.add_argument("--keyword", default="")
    parser.add_argument("--model", default="pointnet2_cls_ssg")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=os.path.join(POINTNET_ROOT, "target_inference_tools", "outputs"))
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--num_category", type=int, default=2)
    parser.add_argument("--num_point", type=int, default=192)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--mode", choices=["yz", "xz"], default="yz")
    parser.add_argument("--eps", type=float, default=0.45)
    parser.add_argument("--min_samples", type=int, default=3)
    parser.add_argument("--min_cluster_points", type=int, default=6)
    parser.add_argument("--merge_dist", type=float, default=0.8)
    parser.add_argument("--dist_threshold", type=float, default=0.9)
    parser.add_argument("--max_missed", type=int, default=10)
    parser.add_argument("--max_points_ratio", type=float, default=2.8)
    parser.add_argument("--max_step_dist", type=float, default=1.1)
    parser.add_argument("--min_track_len", type=int, default=10)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_normals or args.num_category != 2:
        raise ValueError("Radar inference expects XYZ input and two classes")
    if args.window_size < 1 or args.stride < 1:
        raise ValueError("window_size and stride must be positive")
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    os.makedirs(args.output, exist_ok=True)

    input_files = sorted(
        os.path.join(args.input, name)
        for name in os.listdir(args.input)
        if name.endswith("_frames.mat")
    )
    input_files = [path for path in input_files if ("双人" in os.path.basename(path)) or ("三人" in os.path.basename(path))]
    if args.keyword:
        input_files = [path for path in input_files if args.keyword in os.path.basename(path)]
    if not input_files:
        raise RuntimeError("No multi-person processed files found")

    classifier = load_classifier(args.model, args.checkpoint, args.num_category, args.use_normals, device)

    run_index = []
    for file_path in input_files:
        frames, flags = load_frames_and_flags(file_path)
        tracker = MultiObjectTracker(
            dist_threshold=args.dist_threshold,
            max_missed=args.max_missed,
            max_points_ratio=args.max_points_ratio,
        )

        for frame_idx, (frame_points, frame_flag) in enumerate(zip(frames, flags)):
            clusters, _labels = cluster_one_frame(
                frame_points,
                mode=args.mode,
                eps=args.eps,
                min_samples=args.min_samples,
                min_cluster_points=args.min_cluster_points,
            )
            clusters = merge_close_clusters(clusters, mode=args.mode, merge_dist=args.merge_dist)
            tracker.update(clusters, frame_idx, frame_flag)

        tracks = [track for track in tracker.all_tracks() if len(track.history) >= args.min_track_len]
        tracks = refine_tracks(tracks, max_step_dist=args.max_step_dist, min_track_len=args.min_track_len)

        windows = []
        for track in tracks:
            windows.extend(build_track_windows(track, args.window_size, args.stride, args.num_point))

        predictions = infer_windows(classifier, windows, device)
        track_summary = summarize_track_predictions(predictions, args.decision_threshold)

        case_name = os.path.splitext(os.path.basename(file_path))[0]
        case_dir = os.path.join(args.output, case_name)
        os.makedirs(case_dir, exist_ok=True)

        result = {
            "source_file": file_path,
            "num_frames": len(frames),
            "flag_hist": dict(Counter(flags)),
            "num_tracks": len(tracks),
            "num_windows": len(windows),
            "track_summary": track_summary,
            "window_predictions": predictions,
            "params": vars(args),
        }

        with open(os.path.join(case_dir, "inference_summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        run_index.append(
            {
                "source_file": file_path,
                "num_tracks": len(tracks),
                "num_windows": len(windows),
                "top_track": track_summary[0] if track_summary else None,
            }
        )

        print(f"processed: {file_path}")
        print(f"tracks={len(tracks)} windows={len(windows)}")
        if track_summary:
            print(f"top track summary: {track_summary[0]}")

    with open(os.path.join(args.output, "run_index.json"), "w", encoding="utf-8") as f:
        json.dump(run_index, f, indent=2, ensure_ascii=False)

    print(f"saved outputs to: {args.output}")


if __name__ == "__main__":
    main()
