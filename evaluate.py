import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.RadarFallDataLoader import RadarFallDataLoader
from predict import ROOT_DIR, load_model
from training_utils import set_seed
from train_classification import compute_metrics_from_confusion


def main():
    parser = argparse.ArgumentParser('Evaluate a checkpoint on file-held-out radar windows')
    parser.add_argument('--data_root', default=str(ROOT_DIR / 'processed_data'))
    parser.add_argument('--checkpoint', default=str(ROOT_DIR / 'weights/fall_ssg_reference.pth'))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--output', default=str(ROOT_DIR / 'outputs/evaluation.json'))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch_size must be positive')
    set_seed(args.seed)
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else 'cuda')
    model, checkpoint = load_model(args.checkpoint, device)
    pre = checkpoint.get('preprocessing', {})
    if pre.get('weak_scene_labels'):
        parser.error('This evaluator uses frame windows; use the track training evaluator for track models')
    dataset = RadarFallDataLoader(args.data_root, SimpleNamespace(num_point=pre.get('num_point', 192)),
                                 split='test', test_ratio=pre.get('test_ratio', 0.2),
                                 window_size=pre.get('window_size', 15), stride=pre.get('stride', 5),
                                 single_only=pre.get('single_only', True),
                                 legacy_name_filter=pre.get('legacy_name_filter', False))
    confusion = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for points, target in tqdm(DataLoader(dataset, batch_size=args.batch_size), desc='eval'):
            scores, _ = model(points.transpose(2, 1).to(device))
            for label, prediction in zip(target.tolist(), scores.argmax(1).cpu().tolist()):
                confusion[label, prediction] += 1
    result = {'checkpoint': Path(args.checkpoint).name, 'checkpoint_epoch': checkpoint.get('epoch'),
              'seed': args.seed, 'num_windows': len(dataset), 'preprocessing': pre,
              'metrics': compute_metrics_from_confusion(confusion), 'confusion': confusion.tolist(),
              'split': 'sorted files, held-out tail; not a subject-independent evaluation'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
