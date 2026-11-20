import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "models"))

import provider
from training_utils import set_seed, validate_training, validate_datasets
from scene_window_dataset import POINTNET_ROOT, SceneWindowDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone scene-level multi-person training")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model", type=str, default="pointnet2_cls_ssg")
    parser.add_argument("--num_category", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--num_point", type=int, default=192)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--data_root", type=str, default=os.path.join(POINTNET_ROOT, "processed_data"))
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min_segment_len", type=int, default=6)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--include_single", action="store_true", default=False)
    parser.add_argument("--weighted_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampler_power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def inplace_relu(module):
    if module.__class__.__name__.find("ReLU") != -1:
        module.inplace = True


def compute_class_weights(labels, num_class):
    counts = np.bincount(labels, minlength=num_class).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_class * counts)
    weights = np.power(weights, 0.5)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def setup_logger(log_dir, model_name):
    logger = logging.getLogger("SceneLevelTrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(str(log_dir / f"{model_name}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_string(logger, message):
    logger.info(message)
    print(message)


def evaluate(model, loader, num_class, device):
    model.eval()
    mean_correct = []
    class_acc = np.zeros((num_class, 3), dtype=np.float32)
    confusion = np.zeros((num_class, num_class), dtype=np.int32)

    with torch.no_grad():
        for points, target, _meta in tqdm(loader, total=len(loader), desc="eval", leave=False):
            points = points.to(device).transpose(2, 1)
            target = target.to(device)
            pred, _ = model(points)
            pred_choice = pred.data.max(1)[1]
            for cat in np.unique(target.cpu()):
                mask = target == cat
                classacc = pred_choice[mask].eq(target[mask].long()).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[mask].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            for gt, pr in zip(target.cpu().numpy(), pred_choice.cpu().numpy()):
                confusion[int(gt), int(pr)] += 1

    class_acc[:, 2] = class_acc[:, 0] / np.maximum(class_acc[:, 1], 1)
    return {
        "instance_acc": float(np.mean(mean_correct)) if mean_correct else 0.0,
        "class_acc": float(np.mean(class_acc[:, 2])),
        "confusion": confusion,
        "recall_per_class": np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1),
    }


def main(args):
    validate_training(args)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    exp_root = Path(ROOT_DIR) / "log" / "scene_level_classification"
    exp_root.mkdir(parents=True, exist_ok=True)
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = exp_root / (args.log_dir or timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = setup_logger(log_dir, args.model)
    log_string(logger, f"PARAMETERS: {args}")
    log_string(logger, f"device: {device}")

    train_dataset = SceneWindowDataset(
        root=args.data_root,
        args=args,
        split="train",
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        min_segment_len=args.min_segment_len,
        include_single=args.include_single,
        include_multi_only=not args.include_single,
    )
    test_dataset = SceneWindowDataset(
        root=args.data_root,
        args=args,
        split="test",
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        min_segment_len=args.min_segment_len,
        include_single=args.include_single,
        include_multi_only=not args.include_single,
    )

    validate_datasets(train_dataset, test_dataset, args.batch_size)
    drop_last = len(train_dataset) % args.batch_size == 1
    train_loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0, "drop_last": drop_last}
    if args.weighted_sampler:
        class_weights = compute_class_weights(train_dataset.labels, args.num_category)
        class_weights = np.power(class_weights, args.sampler_power)
        class_weights = class_weights / class_weights.mean()
        sample_weights = class_weights[train_dataset.labels]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights.tolist()), len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, **train_loader_kwargs)
        log_string(logger, f"sampler_class_weights: {class_weights.tolist()}")
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model_lib = importlib.import_module(args.model)
    classifier = model_lib.get_model(args.num_category, normal_channel=args.use_normals)
    criterion = model_lib.get_loss()
    classifier.apply(inplace_relu)
    classifier = classifier.to(device)
    criterion = criterion.to(device)

    model_path = os.path.join(ROOT_DIR, "models", f"{args.model}.py")
    if os.path.exists(model_path):
        shutil.copy(model_path, str(exp_dir))
    shutil.copy(os.path.join(BASE_DIR, "train_scene_classification.py"), str(exp_dir))
    shutil.copy(os.path.join(BASE_DIR, "scene_window_dataset.py"), str(exp_dir))

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_metric = -1.0

    for epoch in range(args.epoch):
        log_string(logger, f"Epoch {epoch + 1}/{args.epoch}")
        classifier.train()
        mean_correct = []
        loss_list = []

        for points, target, _meta in tqdm(train_loader, total=len(train_loader), desc="train", leave=False):
            optimizer.zero_grad()
            points_np = points.numpy()
            points_np = provider.random_point_dropout(points_np)
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
            points = torch.tensor(points_np, dtype=torch.float32).transpose(2, 1).to(device)
            target = target.to(device)
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss_list.append(loss.item())

        scheduler.step()
        eval_result = evaluate(classifier, test_loader, args.num_category, device)
        log_string(logger, f"Loss: {float(np.mean(loss_list)):.6f}")
        log_string(logger, f"Train Instance Accuracy: {float(np.mean(mean_correct)):.6f}")
        log_string(logger, f"Test Instance Accuracy: {eval_result['instance_acc']:.6f}, Class Accuracy: {eval_result['class_acc']:.6f}")
        log_string(logger, f"Recall per class: {eval_result['recall_per_class']}")
        log_string(logger, f"Confusion Matrix:\n{eval_result['confusion']}")

        state = {
            "epoch": epoch + 1,
            "instance_acc": eval_result["instance_acc"],
            "class_acc": eval_result["class_acc"],
            "state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(state, checkpoints_dir / "latest.pth")
        if eval_result["class_acc"] >= best_metric:
            best_metric = eval_result["class_acc"]
            torch.save(state, checkpoints_dir / "best.pth")


if __name__ == "__main__":
    main(parse_args())
