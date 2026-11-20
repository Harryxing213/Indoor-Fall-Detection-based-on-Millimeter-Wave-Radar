import argparse
import os

from track_debugger import process_file


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Batch rebuild tracks from processed radar frames")
    parser.add_argument("--input", default=os.path.join(POINTNET_ROOT, "processed_data"))
    parser.add_argument("--output", default=os.path.join(POINTNET_ROOT, "debug_tools", "rebuilt_tracks"))
    parser.add_argument("--keyword", default="")
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
    parser.add_argument("--save-every", dest="save_every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    from glob import glob

    files = sorted(glob(os.path.join(args.input, "*_frames.mat")))
    if args.keyword:
        files = [file_path for file_path in files if args.keyword in os.path.basename(file_path)]
    if not files:
        raise RuntimeError("No matching *_frames.mat files found")

    os.makedirs(args.output, exist_ok=True)
    for file_path in files:
        print(f"processing {file_path}")
        process_file(file_path, args.output, args)
    print(f"rebuilt tracks saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
