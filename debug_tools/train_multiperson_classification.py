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
from track_window_dataset import MultiPersonTrackWindowDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone multi-person fall training")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="pointnet2_cls_ssg")
    parser.add_argument("--num_category", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--num_point", type=int, default=192)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument(
        "--rebuilt_root",
        type=str,
        default=os.path.join(ROOT_DIR, "debug_tools", "rebuilt_tracks"),
    )
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min_track_len", type=int, default=10)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--include_single", action="store_true", default=False)
    parser.add_argument("--min_points_per_window", type=int, default=24)
    parser.add_argument("--save_best_only", action="store_true", default=False)
    parser.add_argument("--weighted_sampler", action="store_true", default=False)
    parser.add_argument("--class_weighted_loss", action="store_true", default=False)
    parser.add_argument("--sampler_power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def inplace_relu(module):
    classname = module.__class__.__name__
    if classname.find("ReLU") != -1:
        module.inplace = True


def setup_logger(log_dir, model_name):
    logger = logging.getLogger("MultiPersonTrain")
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
    class_acc = np.zeros((num_class, 3), dtype=np.float32)
    mean_correct = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for points, target, _meta in tqdm(loader, total=len(loader), desc="eval", leave=False):
            points = points.to(device).transpose(2, 1)
            target = target.to(device)
            pred, _ = model(points)
            pred_choice = pred.data.max(1)[1]

            all_targets.extend(target.cpu().numpy().tolist())
            all_preds.extend(pred_choice.cpu().numpy().tolist())

            for cat in np.unique(target.cpu()):
                mask = target == cat
                classacc = pred_choice[mask].eq(target[mask].long()).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[mask].size()[0])
                class_acc[cat, 1] += 1

            correct = pred_choice.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / np.maximum(class_acc[:, 1], 1)
    confusion = np.zeros((num_class, num_class), dtype=np.int32)
    for target, pred in zip(all_targets, all_preds):
        confusion[target, pred] += 1

    return {
        "instance_acc": float(np.mean(mean_correct)) if mean_correct else 0.0,
        "class_acc": float(np.mean(class_acc[:, 2])),
        "confusion": confusion,
        "recall_per_class": np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1),
    }


def compute_class_weights(labels, num_class):
    counts = np.bincount(labels, minlength=num_class).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_class * counts)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def soften_weights(weights, power):
    softened = np.power(weights, power)
    softened = softened / softened.mean()
    return softened.astype(np.float32)


def main(args):
    validate_training(args)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = Path(ROOT_DIR) / "log" / "multi_person_classification"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir / (args.log_dir or timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = setup_logger(log_dir, args.model)
    log_string(logger, f"PARAMETERS: {args}")
    log_string(logger, f"device: {device}")

    train_dataset = MultiPersonTrackWindowDataset(
        root=args.rebuilt_root,
        args=args,
        split="train",
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        min_track_len=args.min_track_len,
        include_single=args.include_single,
        include_multi_only=not args.include_single,
        min_points_per_window=args.min_points_per_window,
    )
    test_dataset = MultiPersonTrackWindowDataset(
        root=args.rebuilt_root,
        args=args,
        split="test",
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        min_track_len=args.min_track_len,
        include_single=args.include_single,
        include_multi_only=not args.include_single,
        min_points_per_window=args.min_points_per_window,
    )

    validate_datasets(train_dataset, test_dataset, args.batch_size)
    drop_last = len(train_dataset) % args.batch_size == 1
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "drop_last": drop_last,
    }
    if args.weighted_sampler:
        sample_class_weights = compute_class_weights(train_dataset.labels, args.num_category)
        sample_class_weights = soften_weights(sample_class_weights, args.sampler_power)
        sample_weights = sample_class_weights[train_dataset.labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights.tolist()),
            num_samples=len(sample_weights),
            replacement=True,
        )
        log_string(logger, f"sampler_class_weights: {sample_class_weights.tolist()}")
        train_loader = DataLoader(train_dataset, sampler=sampler, **train_loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model_lib = importlib.import_module(args.model)
    classifier = model_lib.get_model(args.num_category, normal_channel=args.use_normals)
    criterion = model_lib.get_loss()
    classifier.apply(inplace_relu)
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    class_weights = None
    if args.class_weighted_loss:
        class_weights = torch.tensor(
            compute_class_weights(train_dataset.labels, args.num_category), dtype=torch.float32, device=device
        )
        log_string(logger, f"class_weights: {class_weights.detach().cpu().numpy().tolist()}")

    model_path = os.path.join(ROOT_DIR, "models", f"{args.model}.py")
    if os.path.exists(model_path):
        shutil.copy(model_path, str(exp_dir))
    shutil.copy(__file__, str(exp_dir))
    shutil.copy(os.path.join(BASE_DIR, "track_window_dataset.py"), str(exp_dir))

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_metric = -1.0

    for epoch in range(args.epoch):
        log_string(logger, f"Epoch {epoch + 1}/{args.epoch}")
        classifier.train()
        loss_list = []
        mean_correct = []

        for points, target, _meta in tqdm(train_loader, total=len(train_loader), desc="train", leave=False):
            optimizer.zero_grad()

            points_np = points.numpy()
            points_np = provider.random_point_dropout(points_np)
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
            points = torch.tensor(points_np, dtype=torch.float32).transpose(2, 1).to(device)
            target = target.to(device)

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat, weight=class_weights)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss_list.append(loss.item())

        scheduler.step()

        train_loss = float(np.mean(loss_list)) if loss_list else 0.0
        train_instance_acc = float(np.mean(mean_correct)) if mean_correct else 0.0
        eval_result = evaluate(classifier, test_loader, args.num_category, device)

        log_string(logger, f"Loss: {train_loss:.6f}")
        log_string(logger, f"Train Instance Accuracy: {train_instance_acc:.6f}")
        log_string(
            logger,
            f"Test Instance Accuracy: {eval_result['instance_acc']:.6f}, Class Accuracy: {eval_result['class_acc']:.6f}",
        )
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

        latest_path = checkpoints_dir / "latest.pth"
        torch.save(state, latest_path)
        selection_metric = eval_result["class_acc"]
        if selection_metric >= best_metric:
            best_metric = selection_metric
            torch.save(state, checkpoints_dir / "best.pth")
        elif not args.save_best_only:
            torch.save(state, checkpoints_dir / f"epoch_{epoch + 1:03d}.pth")

    log_string(logger, "End of training")


if __name__ == "__main__":
    args = parse_args()
    main(args)
