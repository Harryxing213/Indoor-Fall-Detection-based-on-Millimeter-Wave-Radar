# Data notes

## Local folders reviewed

| Folder | Contents |
| --- | --- |
| Ceiling-mounted fall data | 16 raw bathroom MAT files and 51 videos across bathroom / living-room collections |
| Parser | 50 raw MAT files (16 bathroom, 34 living room), 36 processed MAT files and MATLAB sources |
| PointNet cache | 26 processed frame files: 13 single-person and 13 multi-person filenames |

The 26-file cache has 20,464 frames and 482,036 point rows. It contains 706 legacy sentinel rows with XYZ equal to `[100, 100, 100]`. The audit found no non-finite point values. Three single-person files use slash-form flags (`1/0`, `1/1`), totaling 2,666 frames. The frame loader's historical label map skips these flags.

Eleven filenames overlap between the parser's processed folder and the PointNet cache, but none of these pairs are byte-identical. They must not be treated as interchangeable cached files. Historical training used the PointNet cache.

## Window preprocessing and splits

For the 26-file cache at 192 points, window size 15 and stride 5:

| Frame-loader selection | Selected train / validation files | Train / validation windows |
| --- | --- | --- |
| Corrected single-person filter | 10 / 3 | 1,170 / 328 |
| Historical `一人` substring filter | 12 / 3 | 1,492 / 328 |

The historical filter includes two multi-person recordings whose descriptions also contain `一人`. Both are in the training split. The validation windows remain the same for this particular cache. New training uses the corrected filter; the exported reference checkpoint explicitly records the historical setting.

Files are sorted lexicographically, not numerically. Windows from a file stay in the same split. Recordings from the same person may still appear in both splits. No claim of subject-independent generalization follows from these numbers.

The frame loader forms constant-flag segments, merges XYZ points within each window, applies FPS when enough points exist, pads sparse windows by resampling, centers the result and scales it to unit radius. Empty or constant clouds return finite zeros. Each frame contributes points rather than a temporal feature vector.

## Compatibility details

The original type-3 decoder emits a `[100, 100, 100]` row when a radar frame has zero detections. Existing model results used processed data containing these rows. The curated parser preserves this behavior and records it in preprocessing metadata. These coordinates are an empty-frame marker, not a person's position. Removing the marker or normalizing slash flags changes the dataset and requires a new experiment; such changes were not silently applied to the reference results.

The acquisition note states 2.7 m installation height. `ParameterInit.m` uses 2.1 m. The new parser accepts the height explicitly and defaults to the existing 2.1 m value for compatibility. The original calibration cannot be resolved from the files alone.

Track labels are inherited from scene flags. The track and scene experimental loaders retain the old rule of treating suffix `-1` as fall and other flags as non-fall. Use known hyphen-form flags for these branches. Track training also rejects fall-to-non-fall transitions and windows with mixed binary labels. Windows with fewer than 24 merged points return a zero cloud under the retained experimental preprocessing.
