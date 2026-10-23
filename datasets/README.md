# Dataset Guide

This project uses the PSyllabus dataset for piano performance difficulty prediction. The metadata contains 7,901 records. The difficulty label is stored in the `ps` field and ranges from 0 to 10.

## Final Experimental Dataset

The final ten experiments use the frozen common subset `cqt_rhythm_common_v1`:

- Training set: 4,089 pieces
- Validation set: 2,040 pieces
- Test set: 1,557 pieces
- Total: 7,686 pieces

The split files are:

```text
splits/cqt_rhythm_common_v1.json
splits/cqt_rhythm_common_v1_manifest.json
```

This subset inherits the original strict split. It only removes pieces without paired rhythm features or recordings pending verification; it does not reassign samples among train, validation, and test sets.

## Directory Structure

```text
metadata/
  new_clean_data.json        Piece metadata and difficulty labels
  split_audio.json           Original dataset split information
splits/
  cqt_rhythm_common_v1.json  Final frozen split
  *_manifest.json            Per-piece file hashes and membership records
features/
  cqt/                       Per-piece CQT sequences, 88×T at about 5 fps
  rhythm_windows/            Per-piece 9-channel rhythm-window sequences
  cqt_hard_windows.npz       528-dimensional CQT summary features
  rhythm_stats.npz           410-dimensional rhythm summary features
  structure_stats.npz        73-dimensional structure-related statistics
  *_normalization.npz        Normalization parameters computed on training data only
audio/                       Optional original audio, not included by default
```

## Download and Placement

The PSyllabus data and precomputed CQT files are available from the associated Zenodo and MIREX pages:

- https://zenodo.org/records/14794592
- https://music-ir.org/mirex/wiki/2026:Music_Performance_Difficulty_Prediction

Keep the original file names after downloading. Place the CQT `.bin` files in `features/cqt/`. Rhythm-window sequences are extracted from audio and should be placed in `features/rhythm_windows/` together with `index.npz`.

## Notes

- `cqt/`, `rhythm_windows/`, and the original audio are large and excluded by `.gitignore` by default.
- The metadata, split, manifest, and feature caches are protected by hash checks; do not replace them silently.
- Rhythm–CQT alignment currently uses nominal timelines and a shared start point. It has not been verified against waveforms piece by piece.
- Create a new version when regenerating features instead of overwriting files used by the frozen experiments.
