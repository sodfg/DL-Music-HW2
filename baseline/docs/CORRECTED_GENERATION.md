# Corrected generation and evaluation

Current Bar/Pos results are in `runs/corrected_boundary_v2`. Existing six- and three-layer checkpoints were reused; no retraining was required.

## Ending correction

The previous final-bar candidate mask excluded the learned `Bar` boundary while accepting only EOS as a stop signal. This could accumulate notes in the last bar. `boundary-v2` accepts the final Bar as an ending signal and converts it to EOS without creating a 33rd bar. It also blocks duplicate pitches at the same onset while preserving chords and later repeated notes. The per-bar event cap and terminal duration clipping remain.

`generate.py` keeps its historical `legacy` default. Formal runs explicitly pass `--generation-policy boundary-v2`. All 89 corrected Bar/Pos files were generated and evaluated again: 60 Task A, nine Task B, and 20 three-layer samples. The six-layer depth comparison reuses the corrected Task A t10 group. None reached the 128-onset cap.

## Workflow

```bash
python3 baseline/experiments/run_corrected_generation.py --stage plan
CUDA_VISIBLE_DEVICES=0 python3 -u baseline/experiments/run_corrected_generation.py --stage all
python3 baseline/experiments/run_corrected_audio.py --watch-remote \
  --soundfont /path/to/GeneralUser-GS.sf2 --fluidsynth /path/to/fluidsynth
```

The watcher observes an already launched remote job; it does not start training. The default SSH alias is eda_union_sodfg. Plans record source and checkpoint hashes. Existing incompatible plans or partial outputs are not silently overwritten; use a new output directory after source changes.

## Scope

H4/GS use complete generated pieces and 896 complete reference songs, with 64-position bar normalization. SI uses the first 48 seconds, matched piano SoundFont and FluidSynth settings, gain0.025, and 20 fixed reference songs sampled with seed42. SI scales are short3–8s, mid8–15s, long15–48s. Short pieces, clipping, missing values, or source/protocol mismatches cause verification failures rather than silent exclusions.

MIDI-like uses a separate generator and was not affected by the final-Bar mask. Its unchanged samples and matching protocol were verified before reusing results. Representation comparisons include different legal-event and clock constraints; they do not isolate a purely causal representation effect.

Read summary.md/csv/json and postprocess_completed.json. Final aligned Task B files are in runs/task_b_aligned; final Bonus files and reevaluated CA/OA are in runs/bonus_aligned_v2. See SUBMISSION_GUIDE.md for the current package; old assemble_submission.py output is historical.
