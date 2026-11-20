# Verification

The curated Python pipeline was checked locally on CPU. Tested environment: Python 3.12.14, PyTorch 2.15.0, NumPy 2.3.5, h5py 3.16.0, SciPy 1.18.1 and tqdm 4.70.1. The result figure used Matplotlib 3.11.2. Dependency ranges in `requirements.txt` are broader than this one tested environment.

| Check | Outcome |
| --- | --- |
| Python syntax compilation | All distributed Python sources pass |
| CLI help | All 9 main training / data / inference entry points load without hardware dependencies |
| Regression suite | 5 tests pass; gated assignment compared with brute force on 80 small random matrices |
| Exported checkpoints | Both load strictly and produce finite, normalized two-class outputs on CPU |
| Weight preservation | Every exported state tensor equals its original source tensor exactly |
| Real frame loader | Both corrected and historical file selections checked; 48 representative samples have finite 192 × 3 outputs; selected train / validation files are disjoint |
| Reference evaluation | All 328 held-out windows evaluated; confusion `[[177, 84], [6, 61]]` matches the saved record |
| Training execution | Frame, track and scene trainers each completed one CPU epoch on small format fixtures built from local radar points, with labels assigned for execution checks only |
| Resume execution | Frame trainer continued to epoch 2, restored optimizer state and retained both metric-history rows |
| Comparison execution | All three models completed one CPU epoch on a 4-training / 2-validation-window fixture |
| Tracking on real recording | A 1,001-frame multi-person case produced 48 refined segments with unique IDs, using minimum track length 8 |
| Classifier on real multi-person tracks | The same 1,001-frame case produced 37 segments / 89 classified windows with minimum track length 10 |
| NumPy-window prediction | The reference classifier returned finite two-class scores for a real merged 15-frame window |
| Synthetic lifestyle example | SVG / JSON outputs generated successfully and identify themselves as synthetic |
| Measured-result chart | Generated from the saved comparison JSON and visually checked |

The small execution fixtures verify program operation; their labels and one-epoch metrics are not project evaluation results. A fragment count is not a person count. The tracking smoke run does not validate identity accuracy or person-level fall labels.

MATLAB was not available in the verification environment. The conversion wrapper and helper calls were reviewed statically; conversion was not executed in MATLAB. Python verification used the existing processed MATLAB v7.3 files. Raw sensor capture and CUDA execution were not tested.

To repeat the portable regression suite, run `python -m unittest discover -s tests -v` from the repository root. To regenerate the README chart, install `requirements-plots.txt` and run `python model_comparison_tools/plot_recorded_results.py`.
