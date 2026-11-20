# File map

| File / directory | Role |
| --- | --- |
| `matlab/prepare_frames.m` | New configurable type-3 conversion entry; saves Python-readable v7.3 frame cells |
| `matlab/F_CoordinateConversion.m` | Original radar protocol decoding and coordinate transform; type 3 is used here |
| `matlab/ParameterInit.m`, `F_BoxSet.m`, `F_Local_Flag_to_String.m` | Legacy geometry defaults, box geometry and flag conversion |
| `data_utils/RadarFallDataLoader.py` | File split, constant-flag windows, merged XYZ sampling and normalization |
| `data_utils/file_filters.py` | Distinguishes single-person from multi-person filenames before action text |
| `models/pointnet_cls.py`, `pointnet_utils.py` | PointNet classifier and spatial-transform / feature utilities |
| `models/pointnet2_cls_ssg.py`, `pointnet2_cls_msg.py`, `pointnet2_utils.py` | PointNet++ classifiers and grouping / abstraction utilities |
| `provider.py` | Existing point dropout, scale and translation augmentation |
| `train_classification.py` | Frame-window training, weighted loss / sampler, checkpoints and SVG metrics |
| `training_utils.py` | Shared seed and input checks |
| `evaluate.py`, `predict.py` | Radar-specific evaluation and NumPy-window inference |
| `debug_tools/track_debugger.py` | NumPy DBSCAN, cluster merging, Kalman tracking, gated association, track refinement and export |
| `debug_tools/rebuild_all_tracks.py` | Batch track reconstruction |
| `debug_tools/track_window_dataset.py` | Case-disjoint, weakly labeled track windows |
| `debug_tools/train_multiperson_classification.py` | Experimental track-window training |
| `scene_level_tools/` | Whole-scene multi-person baseline |
| `target_inference_tools/run_single_model_on_multiperson.py` | Offline tracking followed by window scores and per-track summaries |
| `model_comparison_tools/run_small_model_comparison.py` | Three-model small-subset experiment and measured-history plots |
| `examples/lifestyle_demo.py` | Synthetic lifestyle illustration, separate from measured results |
| `tests/test_pipeline.py` | Assignment, unique IDs, filename filtering, degenerate clouds and reference-model regression checks |
| `metrics/` | Existing measured history, fresh reference evaluation, weight provenance and hashes |

## Packaging repairs

The curated copy removes unused pose / OpenCV imports from the main trainer, honors CPU selection, accepts a data folder, saves preprocessing metadata and restores optimizer state on resume. Training avoids singleton BatchNorm batches and checks binary XYZ inputs. New defaults use 15-frame windows with stride 5, matching the reference preprocessing.

Single-person filename filtering now excludes multi-person scenes even if their action description contains `一人`. An explicit legacy option retains historical evaluation compatibility. Multi-person `--include_single` filtering and the copied training snapshot were corrected. The experimental weighted-loss branch now retains each model's own criterion, including PointNet regularization.

Gated association uses SciPy's linear assignment instead of factorial permutation enumeration while preserving maximum match count followed by minimum total cost. Refined segments receive unique IDs. Empty-window inference and zero-radius normalization no longer produce sampling errors or NaNs. Inactive track-label ratio arguments were removed because the dataset builds pure-label windows.

## Original branches left outside this package

| Original material | Reason |
| --- | --- |
| Raw MAT, videos, processed frame caches and generated NPZ tracks | Local datasets / derived data, rather than repository source |
| `main2.py` through the later main variants, copies, `jj.py`, `cap_data.py`, `show_data.py`, `pose_api.py`, `new_models/` | Earlier serial / RealSense / pose experiments; hardcoded hardware and unrelated model heads |
| `build_tracks_from_processed_data.py`, `Kalman_Filter.py`, `data_utils/TrackDataLoader.py` | Earlier tracking path; superseded by the more complete debug-tools pipeline |
| ModelNet, ShapeNet, S3DIS loaders; segmentation trainers / tests / models; `visualizer/` | Inherited upstream tasks that are not the radar fall-classification implementation |
| Report figure scripts and projected results | Presentation / illustrative outputs; not validated training metrics |
| Repeated code under `log/`, 130 other checkpoint files, generated plots, IDE and Python caches | Snapshots, historical runs and generated artifacts |
| Empty `fyp.m` | Placeholder with no implementation |

Original folders were left intact. `source_manifest.json` maps copied files to their local source and records later packaging changes by hash. A separate local inventory accompanies the delivery and is not part of the public repository.
