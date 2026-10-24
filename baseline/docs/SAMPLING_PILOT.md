# Small top-k/top-p listening pilot

This exploratory experiment reuses the six-layer Bar/Pos checkpoint without retraining. Each group has three32-bar pieces,T1,seeds123–125.

| Group | Top-k | Top-p | Change |
|---|---:|---:|---|
| A | 32 | 0.95 | reuse the first three historical t10 pieces |
| B | 64 | 0.95 | change only k from A |
| C | 64 | 0.98 | change only p from B |

```bash
python3 baseline/experiments/run_sampling_pilot.py
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_sampling_pilot.py --execute
```

The historical default directory is runs/sampling_pilot. Source/checkpoint identities are verified against the original depth plan, and incomplete/incompatible output is not overwritten. This pilot predates corrected formal generation; do not use its historical scores as current submission results. Use a new --output under runs/_smoke when testing changed sources or settings.

Listen to matched sample numbers across groups, compare the first four bars and middle sections, and record concrete timestamps for repetition, rhythm, melody continuity, and preference. Equal seeds across settings do not create arrangements of the same song. Grammar restrictions mean changing k/p may not change every candidate set.

Pitch4-gram Jaccard overlap includes chord/accompaniment token ordering and ignores timing, velocity, and transposition. It is neither melody identification nor plagiarism detection. Lower overlap does not mean better music. Three samples do not replace formal20-song evaluation. Extend to20 samples and recompute H4/GS/SI only after choosing a useful hypothesis. Remote output requires an explicit download; GitHub push does not synchronize EDA files.
