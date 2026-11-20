import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data_utils.RadarFallDataLoader import farthest_point_sample, pc_normalize
from training_utils import set_seed


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / 'models'))


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model_name = checkpoint.get('model', 'pointnet2_cls_ssg')
    if model_name not in ('pointnet_cls', 'pointnet2_cls_ssg', 'pointnet2_cls_msg'):
        raise ValueError('Unsupported classification model in checkpoint')
    model = importlib.import_module(model_name).get_model(2, normal_channel=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def prepare_points(points, num_point=192):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError('Expected a non-empty [N, 3+] array')
    points = points[:, :3]
    if not np.isfinite(points).all():
        raise ValueError('Point coordinates must be finite')
    if len(points) >= num_point:
        points = farthest_point_sample(points, num_point)
    else:
        indices = np.random.choice(len(points), num_point - len(points), replace=True)
        points = np.concatenate([points, points[indices]], axis=0)
    return pc_normalize(points).astype(np.float32)


def main():
    parser = argparse.ArgumentParser('Predict one merged radar window saved as a NumPy array')
    parser.add_argument('--points', required=True, help='A .npy array with shape [N, 3+]')
    parser.add_argument('--checkpoint', default=str(ROOT_DIR / 'weights/fall_ssg_reference.pth'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else 'cuda')
    model, checkpoint = load_model(args.checkpoint, device)
    num_point = checkpoint.get('preprocessing', {}).get('num_point', 192)
    points = prepare_points(np.load(args.points, allow_pickle=False), num_point)
    with torch.no_grad():
        scores, _ = model(torch.from_numpy(points).T.unsqueeze(0).to(device))
    probabilities = scores.exp().cpu().numpy()[0]
    print(json.dumps({'pred_label': int(probabilities.argmax()),
                      'label_names': ['non_fall', 'fall'],
                      'non_fall_score': float(probabilities[0]),
                      'fall_score': float(probabilities[1])}, indent=2))


if __name__ == '__main__':
    main()
