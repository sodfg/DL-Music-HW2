# Project layout and historical records

| Path | Purpose |
|---|---|
| baseline/ | Core tokenization, preparation, model, training, generation |
| baseline/audio/ | Verified FluidSynth WAV rendering |
| baseline/evaluation/ | H4/GS, SI, and metric summaries |
| baseline/experiments/ | Controlled experiments and submission assembly |
| baseline/representations/ | Alternative MIDI-like tokens and generation |
| baseline/bonus/ | Melody-conditioned encoder-decoder accompaniment |
| baseline/tests/ | Automated tests |
| baseline/docs/ | Usage and methodological limitations |
| runs/ | Private checkpoints, results, diagnostics, build files, and caches |
| output/ | Current submission folder and reports, plus archives |

See RUNS_LAYOUT.md and the local runs/00_README.md for current versus historical paths. Every visible runs directory has a status label. Smoke tests validate execution with small inputs; they are not formal quality results and do not belong in report tables.

Historical source hashes and plans are immutable provenance. Translating documentation/output text can change source hashes even when computation is unchanged. Do not rewrite historical plans to bypass identity checks. Inspect/report modes can show historical records, but new training/generation/evaluation should use a new output if the runner detects changed sources. Archived builders retain original paths and are not guaranteed to run directly from their archive location.

Reorganization preserved original files and current model/audio sources. It did not retrain models, regenerate music, or alter formal scores. Report rendering still uses runs/report_build/fonts.conf. Current report construction is under runs/report_build_v7.

Local changes do not automatically synchronize EDA Union. Source-only rsync is separate from Git push; do not overwrite remote environments, datasets, checkpoints, or active processes. Git staging, commits, pushes, and branch changes remain author-controlled.
