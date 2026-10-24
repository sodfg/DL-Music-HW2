# Model depth: three versus six layers

This experiment changes decoder depth while keeping the Bar/Pos representation and training token budget fixed.

| Setting | Six layers | Three layers |
|---|---:|---:|
| Parameters | 19,952,128 | 10,494,976 |
| Width / heads / dropout | 512 / 8 / 0.1 | same |
| Context / batch | 1024 / 8 | same |
| Steps / learning rate | 20000 / 0.0003 | same |
| Training seed | 42 | 42 |
| Checkpoint | verified step20000 best.pt | step20000 last.pt |

Both use AdamW, weight decay0.01, warmup200 and cosine decay, all896 songs, and no held-out validation set. Song/crop sampling uses a separate Python RNG, making its order independent of model initialization. Initialization and dropout randomness are not identical between architectures. Changing depth also changes parameter count.

## Run

```bash
python3 baseline/experiments/run_depth_experiment.py --stage plan
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_depth_experiment.py --stage all
```

Run long jobs in tmux after checking nvidia-smi. Ctrl+B then D detaches; Ctrl+C while tailing stops the viewer only. Use --resume only after inspecting an interrupted checkpoint. Completed matching runs are retained, and incompatible source identities require a new --output.

Original model/provenance records remain in runs/depth_3_vs_6. Its old Bar/Pos samples and scores are historical. Current formal generation and scores are in runs/corrected_boundary_v2, using identical boundary-v2 constraints and T1/k32/p0.95/seeds123–142,20 samples of32 bars per model.

## SI

```bash
python3 baseline/experiments/run_depth_si.py
python3 baseline/experiments/run_depth_si.py --execute \
  --si-python .venv-si/bin/python --fluidsynth /path/to/fluidsynth \
  --soundfont /path/to/GeneralUser-GS.sf2
```

The original SI study under depth_3_vs_6/si48 retains reference/protocol records used for verification. New formal Bar/Pos SI is listed in corrected_boundary_v2/summary.md. All SI groups use20 songs and the first48 seconds. Real reference songs were sampled with seed42 from895 eligible songs. Keep source studies when backing up symbolic links.

Fixed steps approximate a fixed training-token budget, not fixed FLOPs. Historical log elapsed times and wrapper process wall times have different definitions, so they cannot establish a speedup ratio. A single training seed and training-probe loss cannot establish statistical significance or generalization. H4/GS/SI measure different properties and cannot identify an overall quality winner.

GitHub push does not synchronize the EDA working folder. Preview a source-only rsync before separately updating it; do not transfer local environments or overwrite remote datasets/checkpoints.
