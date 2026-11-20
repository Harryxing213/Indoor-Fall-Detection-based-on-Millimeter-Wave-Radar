import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser('Plot the recorded small-subset comparison')
    parser.add_argument('--input', default=str(ROOT_DIR / 'metrics/small_model_comparison_results.json'))
    parser.add_argument('--output', default=str(ROOT_DIR / 'assets/model_comparison.png'))
    args = parser.parse_args()
    results = json.loads(Path(args.input).read_text())['results']
    fig, ax = plt.subplots(figsize=(9.0, 4.8), layout='constrained')
    positions = np.arange(len(results))
    colors = ['#36678d', '#df9844', '#549482']
    for offset, (key, label) in enumerate([('accuracy', 'Accuracy'), ('recall', 'Fall recall'), ('f1', 'F1')]):
        values = [row['best']['val'][key] * 100 for row in results]
        bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=label, color=colors[offset])
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=3, fontsize=9)
    ax.set_xticks(positions, [row['display'] for row in results])
    ax.set_ylim(0, 103)
    ax.set_ylabel('Score (%)')
    ax.set_title('Recorded model comparison', loc='left', weight='bold', pad=24)
    ax.text(0, 1.025, '15 epochs · 384 training / 256 validation windows · historical filename filter',
            transform=ax.transAxes, fontsize=9, color='#555555')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.16)
    ax.legend(frameon=False, ncols=3, loc='upper right')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
