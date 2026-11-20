import argparse
import importlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(POINTNET_ROOT, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

sys.path.append(POINTNET_ROOT)
sys.path.append(MODELS_DIR)

import provider  # noqa: E402
from data_utils.RadarFallDataLoader import RadarFallDataLoader  # noqa: E402


@dataclass
class LoaderArgs:
    num_point: int = 192
    use_uniform_sample: bool = False
    use_normals: bool = False
    process_data: bool = False


MODEL_DISPLAY = {
    "pointnet_cls": "PointNet",
    "pointnet2_cls_ssg": "PointNet++ SSG",
    "pointnet2_cls_msg": "PointNet++ MSG",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inplace_relu(module):
    if module.__class__.__name__.find("ReLU") != -1:
        module.inplace = True


def balanced_subset_indices(dataset, max_samples: int, seed: int):
    by_label = {0: [], 1: []}
    for idx, sample in enumerate(dataset.samples):
        label = int(sample["label"])
        by_label.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    for indices in by_label.values():
        rng.shuffle(indices)
    per_class = max_samples // 2
    selected = by_label.get(0, [])[:per_class] + by_label.get(1, [])[:per_class]
    if len(selected) < max_samples:
        remaining = [i for i in range(len(dataset)) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])
    rng.shuffle(selected)
    return selected


def compute_metrics(confusion: np.ndarray):
    tn = int(confusion[0, 0])
    fp = int(confusion[0, 1])
    fn = int(confusion[1, 0])
    tp = int(confusion[1, 1])
    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model, loader, criterion, device, num_class=2):
    model.eval()
    total_loss = []
    confusion = np.zeros((num_class, num_class), dtype=np.int32)
    with torch.no_grad():
        for points, target in loader:
            points = points.to(device).transpose(2, 1)
            target = target.to(device)
            pred, trans_feat = model(points)
            loss = criterion(pred, target.long(), trans_feat)
            total_loss.append(float(loss.item()))
            pred_choice = pred.data.max(1)[1]
            for gt, pr in zip(target.cpu().numpy(), pred_choice.cpu().numpy()):
                confusion[int(gt), int(pr)] += 1
    metrics = compute_metrics(confusion)
    metrics["loss"] = float(np.mean(total_loss)) if total_loss else 0.0
    metrics["confusion"] = confusion.tolist()
    return metrics


def train_one_model(model_name, train_loader, val_loader, args, device):
    model_lib = importlib.import_module(model_name)
    classifier = model_lib.get_model(2, normal_channel=False)
    criterion = model_lib.get_loss()
    classifier.apply(inplace_relu)
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.decay_rate,
    )

    class_weights = None
    if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'indices'):
        base_dataset = train_loader.dataset.dataset
        subset_indices = train_loader.dataset.indices
        labels = [int(base_dataset.samples[i]['label']) for i in subset_indices]
        counts = {0: labels.count(0), 1: labels.count(1)}
        total = max(sum(counts.values()), 1)
        weights = [total / (2 * max(counts.get(i, 0), 1)) for i in range(2)]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        train_losses = []
        train_confusion = np.zeros((2, 2), dtype=np.int32)
        for points, target in train_loader:
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

            train_losses.append(float(loss.item()))
            pred_choice = pred.data.max(1)[1]
            for gt, pr in zip(target.cpu().numpy(), pred_choice.cpu().numpy()):
                train_confusion[int(gt), int(pr)] += 1

        train_metrics = compute_metrics(train_confusion)
        train_metrics["loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        val_metrics = evaluate(classifier, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        current_key = (val_metrics["accuracy"], val_metrics["f1"], -val_metrics["loss"])
        best_key = None if best is None else (best["val"]["accuracy"], best["val"]["f1"], -best["val"]["loss"])
        if best is None or current_key > best_key:
            best = record
        print(
            f"{MODEL_DISPLAY[model_name]} epoch {epoch}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}"
        )
    return {"model": model_name, "display": MODEL_DISPLAY[model_name], "history": history, "best": best}


def metric_series(results, metric_name, split):
    series = {}
    for result in results:
        series[result["display"]] = [row[split][metric_name] for row in result["history"]]
    return series


def svg_line_chart(title, x_label, y_label, series: Dict[str, List[float]], output_path, y_min=None, y_max=None):
    width, height = 1100, 680
    left, right, top, bottom = 110, 60, 80, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#222222", "#1f5a99", "#b22222", "#5a7d2b"]
    max_len = max(len(v) for v in series.values()) if series else 1
    values = [x for vals in series.values() for x in vals]
    if y_min is None:
        y_min = min(values) if values else 0.0
    if y_max is None:
        y_max = max(values) if values else 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad

    def px(epoch_idx):
        if max_len == 1:
            return left + plot_w / 2
        return left + (epoch_idx / (max_len - 1)) * plot_w

    def py(value):
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>.title{font:700 28px 'Times New Roman',serif}.label{font:18px 'Times New Roman',serif}.small{font:15px 'Times New Roman',serif}</style>",
        f'<text x="{width/2}" y="42" text-anchor="middle" class="title">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<text x="{width/2}" y="{height-35}" text-anchor="middle" class="label">{x_label}</text>',
        f'<text x="32" y="{top+plot_h/2}" text-anchor="middle" class="label" transform="rotate(-90 32 {top+plot_h/2})">{y_label}</text>',
    ]
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = py(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="small">{value:.2f}</text>')
    for i in range(max_len):
        x = px(i)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle" class="small">{i+1}</text>')

    for idx, (name, vals) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.8"/>')
        for i, v in enumerate(vals):
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" fill="{color}"/>')
        legend_y = 82 + idx * 26
        parts.append(f'<line x1="{width-250}" y1="{legend_y}" x2="{width-205}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-195}" y="{legend_y+5}" class="small">{name}</text>')
    parts.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_results(results, args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "small_model_comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    table_path = os.path.join(OUTPUT_DIR, "small_model_comparison_table.csv")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("Model,Best epoch,Training loss,Validation loss,Training accuracy,Validation accuracy,Precision,Recall,F1-score\n")
        for result in results:
            best = result["best"]
            f.write(
                f"{result['display']},{best['epoch']},{best['train']['loss']:.4f},{best['val']['loss']:.4f},"
                f"{best['train']['accuracy']:.4f},{best['val']['accuracy']:.4f},"
                f"{best['val']['precision']:.4f},{best['val']['recall']:.4f},{best['val']['f1']:.4f}\n"
            )

    svg_line_chart(
        "Training Loss of Candidate Models",
        "Epoch",
        "Training loss",
        metric_series(results, "loss", "train"),
        os.path.join(OUTPUT_DIR, "training_loss.svg"),
    )
    svg_line_chart(
        "Validation Loss of Candidate Models",
        "Epoch",
        "Validation loss",
        metric_series(results, "loss", "val"),
        os.path.join(OUTPUT_DIR, "validation_loss.svg"),
    )
    svg_line_chart(
        "Training Accuracy of Candidate Models",
        "Epoch",
        "Training accuracy",
        metric_series(results, "accuracy", "train"),
        os.path.join(OUTPUT_DIR, "training_accuracy.svg"),
        y_min=0,
        y_max=1,
    )
    svg_line_chart(
        "Validation Accuracy of Candidate Models",
        "Epoch",
        "Validation accuracy",
        metric_series(results, "accuracy", "val"),
        os.path.join(OUTPUT_DIR, "validation_accuracy.svg"),
        y_min=0,
        y_max=1,
    )
    svg_line_chart(
        "Validation Precision of Candidate Models",
        "Epoch",
        "Precision",
        metric_series(results, "precision", "val"),
        os.path.join(OUTPUT_DIR, "validation_precision.svg"),
        y_min=0,
        y_max=1,
    )
    svg_line_chart(
        "Validation Recall of Candidate Models",
        "Epoch",
        "Recall",
        metric_series(results, "recall", "val"),
        os.path.join(OUTPUT_DIR, "validation_recall.svg"),
        y_min=0,
        y_max=1,
    )
    svg_line_chart(
        "Validation F1-score of Candidate Models",
        "Epoch",
        "F1-score",
        metric_series(results, "f1", "val"),
        os.path.join(OUTPUT_DIR, "validation_f1.svg"),
        y_min=0,
        y_max=1,
    )
    return json_path, table_path


def parse_args():
    parser = argparse.ArgumentParser("small model comparison")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--max_train", type=int, default=384)
    parser.add_argument("--max_val", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--models", nargs="+", default=["pointnet_cls", "pointnet2_cls_ssg", "pointnet2_cls_msg"])
    parser.add_argument("--legacy_name_filter", action="store_true", help="reproduce historical file filtering")
    parser.add_argument("--data_root", default=os.path.join(POINTNET_ROOT, "processed_data"))
    parser.add_argument("--output", default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = args.output
    if args.batch_size < 2 or args.max_train < 2 or args.max_val < 1:
        raise ValueError("Use batch_size/max_train >= 2 and max_val >= 1")
    set_seed(args.seed)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    loader_args = LoaderArgs(num_point=192)
    train_dataset = RadarFallDataLoader(
        root=args.data_root,
        args=loader_args,
        split="train",
        task="fall_binary",
        window_size=15,
        stride=5,
        single_only=True,
        legacy_name_filter=args.legacy_name_filter,
    )
    val_dataset = RadarFallDataLoader(
        root=args.data_root,
        args=loader_args,
        split="test",
        task="fall_binary",
        window_size=15,
        stride=5,
        single_only=True,
        legacy_name_filter=args.legacy_name_filter,
    )
    train_indices = balanced_subset_indices(train_dataset, min(args.max_train, len(train_dataset)), args.seed)
    val_indices = balanced_subset_indices(val_dataset, min(args.max_val, len(val_dataset)), args.seed + 1)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_labels = [int(train_dataset.samples[i]['label']) for i in train_indices]
    class_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    sample_weights = [1.0 / max(class_counts[int(label)], 1) for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=len(train_subset) % args.batch_size == 1)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    results = []
    for model_name in args.models:
        print(f"\n=== Training {MODEL_DISPLAY.get(model_name, model_name)} ===")
        results.append(train_one_model(model_name, train_loader, val_loader, args, device))

    json_path, table_path = write_results(results, args)
    print("\nSaved outputs:")
    print(json_path)
    print(table_path)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
