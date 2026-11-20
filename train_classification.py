import os
import sys
import torch
import numpy as np

from training_utils import set_seed, validate_training, validate_datasets


import datetime
import logging
import provider
import importlib
import shutil
import argparse
from collections import Counter

from pathlib import Path
from tqdm import tqdm
from data_utils.RadarFallDataLoader import RadarFallDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def compute_metrics_from_confusion(confusion):
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
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def save_metric_history_csv(history, save_path):
    header = [
        'epoch',
        'train_loss', 'val_loss',
        'train_accuracy', 'val_accuracy',
        'train_precision', 'val_precision',
        'train_recall', 'val_recall',
        'train_f1', 'val_f1',
    ]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in history:
            values = [
                row['epoch'],
                row['train']['loss'], row['val']['loss'],
                row['train']['accuracy'], row['val']['accuracy'],
                row['train']['precision'], row['val']['precision'],
                row['train']['recall'], row['val']['recall'],
                row['train']['f1'], row['val']['f1'],
            ]
            f.write(','.join(str(v) for v in values) + '\n')


def save_line_chart_svg(history, save_path, title, train_key, val_key, y_label, y_min=None, y_max=None):
    if not history:
        return
    width, height = 1100, 680
    left, right, top, bottom = 115, 65, 82, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    epochs = [row['epoch'] for row in history]
    train_values = [row['train'][train_key] for row in history]
    val_values = [row['val'][val_key] for row in history]
    values = train_values + val_values
    if y_min is None:
        y_min = min(values)
    if y_max is None:
        y_max = max(values)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    pad = max((y_max - y_min) * 0.08, 0.02)
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad
    min_epoch, max_epoch = min(epochs), max(epochs)
    if min_epoch == max_epoch:
        max_epoch = min_epoch + 1

    def px(epoch):
        return left + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_w

    def py(value):
        return top + (y_max - value) / (y_max - y_min) * plot_h

    def make_points(vals):
        return ' '.join(f'{px(e):.1f},{py(v):.1f}' for e, v in zip(epochs, vals))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>.title{font:700 28px 'Times New Roman',serif}.label{font:18px 'Times New Roman',serif}.small{font:15px 'Times New Roman',serif}</style>",
        f'<text x="{width/2}" y="42" text-anchor="middle" class="title">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222" stroke-width="2"/>',
        f'<text x="{width/2}" y="{height-35}" text-anchor="middle" class="label">Epoch</text>',
        f'<text x="34" y="{top+plot_h/2}" text-anchor="middle" class="label" transform="rotate(-90 34 {top+plot_h/2})">{y_label}</text>',
    ]
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = py(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="small">{value:.2f}</text>')
    tick_count = min(6, len(epochs))
    for i in range(tick_count):
        idx = round(i * (len(epochs) - 1) / max(tick_count - 1, 1))
        e = epochs[idx]
        x = px(e)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" class="small">{e}</text>')
    parts.append(f'<polyline points="{make_points(train_values)}" fill="none" stroke="#1f5a99" stroke-width="3"/>')
    parts.append(f'<polyline points="{make_points(val_values)}" fill="none" stroke="#b22222" stroke-width="3"/>')
    parts.append(f'<line x1="{width-250}" y1="86" x2="{width-205}" y2="86" stroke="#1f5a99" stroke-width="3"/>')
    parts.append(f'<text x="{width-195}" y="91" class="small">Training</text>')
    parts.append(f'<line x1="{width-250}" y1="114" x2="{width-205}" y2="114" stroke="#b22222" stroke-width="3"/>')
    parts.append(f'<text x="{width-195}" y="119" class="small">Validation</text>')
    parts.append('</svg>')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


def save_confusion_matrix_svg(confusion, save_path, title='Confusion Matrix'):
    width, height = 760, 620
    labels = ['Non-fall', 'Fall']
    max_value = max(int(confusion.max()), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>.title{font:700 28px 'Times New Roman',serif}.label{font:18px 'Times New Roman',serif}.cell{font:700 28px 'Times New Roman',serif}.small{font:15px 'Times New Roman',serif}</style>",
        f'<text x="{width/2}" y="42" text-anchor="middle" class="title">{title}</text>',
        f'<text x="430" y="92" text-anchor="middle" class="label">Predicted label</text>',
        f'<text x="72" y="350" text-anchor="middle" class="label" transform="rotate(-90 72 350)">True label</text>',
    ]
    x0, y0, cell = 250, 150, 180
    for j, label in enumerate(labels):
        parts.append(f'<text x="{x0 + j*cell + cell/2}" y="132" text-anchor="middle" class="label">{label}</text>')
    for i, label in enumerate(labels):
        parts.append(f'<text x="{x0-35}" y="{y0 + i*cell + cell/2 + 6}" text-anchor="end" class="label">{label}</text>')
    for i in range(2):
        for j in range(2):
            value = int(confusion[i, j])
            intensity = value / max_value
            shade = int(245 - 120 * intensity)
            color = f'rgb({shade},{shade},{shade})'
            x = x0 + j * cell
            y = y0 + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#111" stroke-width="2"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+10}" text-anchor="middle" class="cell">{value}</text>')
    metrics = compute_metrics_from_confusion(confusion)
    parts.append(f'<text x="{width/2}" y="560" text-anchor="middle" class="small">Accuracy={metrics["accuracy"]:.3f}, Precision={metrics["precision"]:.3f}, Recall={metrics["recall"]:.3f}, F1={metrics["f1"]:.3f}</text>')
    parts.append('</svg>')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


def save_training_figures(history, fig_dir, final_confusion):
    fig_dir.mkdir(exist_ok=True)
    save_metric_history_csv(history, str(fig_dir / 'metrics_history.csv'))
    save_line_chart_svg(history, str(fig_dir / 'loss_curve.svg'), 'Training and Validation Loss', 'loss', 'loss', 'Loss')
    save_line_chart_svg(history, str(fig_dir / 'accuracy_curve.svg'), 'Training and Validation Accuracy', 'accuracy', 'accuracy', 'Accuracy', y_min=0, y_max=1)
    save_line_chart_svg(history, str(fig_dir / 'precision_curve.svg'), 'Training and Validation Precision', 'precision', 'precision', 'Precision', y_min=0, y_max=1)
    save_line_chart_svg(history, str(fig_dir / 'recall_curve.svg'), 'Training and Validation Recall', 'recall', 'recall', 'Recall', y_min=0, y_max=1)
    save_line_chart_svg(history, str(fig_dir / 'f1_curve.svg'), 'Training and Validation F1-score', 'f1', 'f1', 'F1-score', y_min=0, y_max=1)
    save_confusion_matrix_svg(final_confusion, str(fig_dir / 'confusion_matrix.svg'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='classification model module')
    parser.add_argument('--num_category', default=2, type=int,  help='binary fall classification: 2 classes')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--window_size', type=int, default=15, help='number of consecutive frames in one sample window')
    parser.add_argument('--stride', type=int, default=5, help='window stride when building training samples')
    parser.add_argument('--min_segment_len', type=int, default=5, help='minimum usable segment length before windowing')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of files used for held-out split')
    parser.add_argument('--data_mode', type=str, default='single', choices=['single', 'all'], help='single uses only one-person files; all uses both one-person and multi-person files')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from log_dir/latest checkpoint')
    parser.add_argument('--balanced_sampling', action=argparse.BooleanOptionalAction, default=True, help='use weighted sampling for class balance')
    parser.add_argument('--data_root', default=os.path.join(ROOT_DIR, 'processed_data'))
    parser.add_argument('--output_root', default=os.path.join(ROOT_DIR, 'log', 'classification'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--legacy_name_filter', action='store_true', help='reproduce historical substring filtering')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, criterion=None, num_class = 2, class_weights=None):
    classifier = model.eval()
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    device = next(model.parameters()).device
    all_targets = []
    all_preds = []
    loss_list = []

    with torch.no_grad():
        for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):


            points, target= points.to(device), target.to(device)

            points = points.transpose(2, 1)
            pred, tran = classifier(points)
            if criterion is not None:
                loss = criterion(pred, target.long(), tran, weight=class_weights)
                loss_list.append(loss.item())
            pred_choice = pred.data.max(1)[1]
            all_targets.extend(target.cpu().numpy().tolist())
            all_preds.extend(pred_choice.cpu().numpy().tolist())

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long()).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1

            correct = pred_choice.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / np.maximum(class_acc[:, 1], 1)
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    confusion = np.zeros((num_class, num_class), dtype=np.int32)
    for gt, pred_label in zip(all_targets, all_preds):
        confusion[int(gt), int(pred_label)] += 1
    metrics = compute_metrics_from_confusion(confusion)
    metrics['loss'] = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
    metrics['instance_acc'] = float(instance_acc)
    metrics['class_acc'] = float(class_acc)

    return metrics, confusion


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    validate_training(args)
    set_seed(args.seed)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path(args.output_root)
    exp_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_root
    single_only = args.data_mode == 'single'
    log_string(f'Data mode: {args.data_mode} (single_only={single_only})')

    train_dataset = RadarFallDataLoader(
        root=data_path,
        args=args,
        split='train',
        task='fall_binary',
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        min_segment_len=args.min_segment_len,
        single_only=single_only,
        legacy_name_filter=args.legacy_name_filter,
    )
    test_dataset = RadarFallDataLoader(
        root=data_path,
        args=args,
        split='test',
        task='fall_binary',
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        min_segment_len=args.min_segment_len,
        single_only=single_only,
        legacy_name_filter=args.legacy_name_filter,
    )
    validate_datasets(train_dataset, test_dataset, args.batch_size)
    drop_last = len(train_dataset) % args.batch_size == 1
    train_label_counts = Counter(int(sample['label']) for sample in train_dataset.samples)
    log_string(f'Train label counts: {dict(train_label_counts)}')
    class_weights_np = np.ones(num_class := args.num_category, dtype=np.float32)
    total_train = max(sum(train_label_counts.values()), 1)
    for cls_idx in range(num_class):
        cls_count = max(train_label_counts.get(cls_idx, 0), 1)
        class_weights_np[cls_idx] = total_train / (num_class * cls_count)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device if 'device' in locals() else 'cpu')

    if args.balanced_sampling:
        sample_weights = np.array([class_weights_np[int(sample['label'])] for sample in train_dataset.samples], dtype=np.float64)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=drop_last)
    else:
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=drop_last)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print("device: ", device)
    class_weights = class_weights.to(device)
    log_string(f'Class weights: {class_weights.detach().cpu().numpy().tolist()}')

    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy(os.path.join(ROOT_DIR, 'models', args.model + '.py'), str(exp_dir))
    shutil.copy(os.path.join(ROOT_DIR, 'models', 'pointnet2_utils.py'), str(exp_dir))
    shutil.copy(os.path.join(ROOT_DIR, 'models', 'pointnet_utils.py'), str(exp_dir))
    shutil.copy(__file__, str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    classifier = classifier.to(device)
    criterion  = criterion.to(device)

    resume_path = checkpoints_dir.joinpath('latest.pth')
    checkpoint = None
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(resume_path)
    if args.resume and resume_path.exists():
        checkpoint = torch.load(str(resume_path), map_location=device, weights_only=True)
        start_epoch = int(checkpoint.get('epoch', 0))
        classifier.load_state_dict(checkpoint['state_dict'])
        log_string(f'Resumed model from {resume_path}')
    else:
        log_string('Starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_epoch = 0
    global_step = 0
    metric_history = checkpoint.get('history', [checkpoint['metrics']]) if checkpoint is not None else []
    final_confusion = np.zeros((num_class, num_class), dtype=np.int32)

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d/%d:' % (epoch + 1, args.epoch))
        loss_list = []

        mean_correct = []
        train_confusion = np.zeros((num_class, num_class), dtype=np.int32)
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.to(device), target.to(device)

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat, weight=class_weights)
            loss_list.append(loss.item())
            pred_choice = pred.data.max(1)[1]
            for gt, pred_label in zip(target.cpu().numpy(), pred_choice.cpu().numpy()):
                train_confusion[int(gt), int(pred_label)] += 1

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            loss.backward()
            optimizer.step()

        train_loss = np.mean(loss_list)
        log_string('Loss: %f' % train_loss)

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        train_metrics = compute_metrics_from_confusion(train_confusion)
        train_metrics['loss'] = float(train_loss)

        val_metrics, final_confusion = test(classifier, testDataLoader, criterion=criterion, num_class=num_class, class_weights=class_weights)
        log_string('Validation Loss: %f' % val_metrics['loss'])
        log_string('Validation Accuracy: %f, Precision: %f, Recall: %f, F1-score: %f' % (
            val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1']
        ))
        log_string('Learning Rate: %f' % optimizer.param_groups[0]['lr'])
        log_string('Confusion Matrix:\n%s' % final_confusion)

        metric_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
        })
        save_training_figures(metric_history, exp_dir.joinpath('figures'), final_confusion)

        state = {
            'state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'metrics': metric_history[-1],
            'history': metric_history,
            'model': args.model,
            'num_category': args.num_category,
            'preprocessing': {'num_point': args.num_point, 'window_size': args.window_size,
                              'stride': args.stride, 'single_only': single_only, 'test_ratio': args.test_ratio,
                              'legacy_name_filter': args.legacy_name_filter},
            'args': vars(args),
        }

        log_string('Saving Dict...')
        savepath = str(checkpoints_dir) + '/latest.pth'
        torch.save(state, savepath)
        global_step += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
