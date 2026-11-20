# Result provenance

## Measured records included

- `metrics/small_model_comparison_results.json`: the existing 15-epoch, 384 / 256-window comparison of PointNet, SSG and MSG. The entire history is retained unchanged. Best epochs use the original validation accuracy / F1 / loss ordering. No comparison checkpoints were saved by that script.
- `metrics/single_person_ssg_history.csv`: three completed epochs from `log/classification/report_small_pipeline`. Its old name reflects the intended single-person setting; the original filename filter also selected two multi-person recordings.
- `metrics/reference_evaluation.json`: a fresh CPU evaluation of the exported epoch-3 checkpoint, seed 42, 328 held-out windows. It reproduced the recorded confusion matrix `[[177, 84], [6, 61]]`.
- `weights/fall_ssg_reference.pth`: the corresponding binary SSG state. The log configured a longer run, but the saved checkpoint contains epoch 3. Planned epochs are not completed epochs.
- `weights/multi_person_ssg_smoke.pth`: epoch 2 from `best_soft_sampler_e6`, an experimental track run. Its saved instance/class accuracy fields are the original averages over batches. They are not independent person-level evaluation metrics. The saved state is provided for experiment inspection, not as the winning model from the three-model comparison.

The reference model's high fall recall is accompanied by 84 false positives among 261 non-fall windows. Window-level validation does not establish online event detection performance.

## Local material excluded from benchmark claims

- `generate_projected_full_scale_results.py` builds projected 2,000-epoch curves from fixed targets and random noise. Those curves are not training logs.
- The report-style accuracy, loss and recall scripts generate shaped / noisy demonstration curves. The report confusion-matrix script uses a hand-set `[[202, 23], [10, 190]]` matrix, which gives 92.3%; it is not the reference checkpoint's evaluation.
- Image-guided radar / trajectory figures contain manually specified geometry and jitter. Video frame overlays in report scripts are illustrative and do not provide synchronized per-person ground truth.
- Root `best_model.pth` and the 84 / 96 / 128 / 192 variants have four-class heads. `40.pth`, `66.pth` and `467.pth` belong to the earlier three-class / pose branch. Their accuracy fields do not describe binary fall classification.
- `examples/lifestyle_demo.py` is retained only as an explicitly synthetic illustration. It uses hand-defined dwell times, paths and anomaly scores.

## Model exports

Both distributed files preserve every source state-dictionary tensor exactly. They contain model and preprocessing metadata but omit optimizer state and machine-specific arguments. `metrics/models.sha256` lists their current hashes; `metrics/model_sources.json` records both original and exported hashes. No additional training was used to change these weights during packaging.
