import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_training(args):
    if args.use_normals or args.num_category != 2:
        raise ValueError('Radar datasets return XYZ points and two classes; do not use normals')
    if args.model not in ('pointnet_cls', 'pointnet2_cls_ssg', 'pointnet2_cls_msg'):
        raise ValueError('Choose one of the three supplied classification models')
    if args.num_point < 128 or args.batch_size < 2:
        raise ValueError('Use num_point >= 128 and batch_size >= 2 for these models')
    if args.window_size < 1 or args.stride < 1 or args.epoch < 1:
        raise ValueError('window_size, stride and epoch must be positive')
    if not 0 < args.test_ratio < 1:
        raise ValueError('test_ratio must be between 0 and 1')


def validate_datasets(train_dataset, val_dataset, batch_size):
    if len(train_dataset) < 2 or len(val_dataset) < 1:
        raise ValueError('Need at least two training windows and one validation window')

