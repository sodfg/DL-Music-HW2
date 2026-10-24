# Submission guide

[Google Drive](https://drive.google.com/drive/folders/1NRQpT-gkLJNPZY7wvyQdwWZ_qVDI4K6x?usp=sharing) and [GitHub](https://github.com/sodfg/DL-Music-HW2) are recorded on the report cover. Link permissions still need author verification.

## Upload folder

Use output/UPLOAD_HW2_B12901157 rather than the complete repository.

| Task | MIDI | WAV | Selection/source |
|---|---:|---:|---|
| A | 20 | 20 | t12, corrected_boundary_v2/basic_a/t12 |
| B | 9 | 9 | task_b_aligned; preferred001=t08,002=t10,003=t08 |
| Bonus | 10 | 10 | bonus_aligned_v2,16annotated bars each |
| Total | 39 | 39 | generation_results/ |

The package also contains code, four inference checkpoints, report PDF/PPTX, metrics, licensed SoundFont, and verification records. Optimizer state is removed from exported models and every weight tensor is compared against its source. Training/preprocessed data, course prompts, caches, environments, Git history, and other researchers' neural checkpoints are excluded from this package. Obtain course data separately when reproducing preparation or evaluation.

The course-provided POP909 version is not redistributed. Exact retraining and full
reference-score reproduction require access to it; upstream POP909 is not a
drop-in replacement. Task A pretrained inference needs no dataset. See the
[data access guide](DATA_ACCESS.md) for supported paths and limitations.

## Fixes reflected in the report

- Basic final-bar masking and duplicate-onset handling were corrected with boundary-v2. Existing checkpoints were reused to regenerate/reassess89Bar/Pos samples. Historical generated scores are not mixed into current tables.
- Task B tick mapping now aligns8prompt+24continuation bars. Performance events remain within0.05ms of source timing.
- Bonus export follows16annotated bars with variable beat counts. Each annotated beat is a display quarter; the original score denominator is not inferred. Timing changes are below0.027ms, with sub-millisecond terminal-beat extension preserving note-offs. All ten final MIDI files were reevaluated because tiny changes can affect duration/IOI bins.
- Submission WAVs use gain1.0,44100Hz stereo PCM16,zero clipping,highest peak0.7324. SI inputs remain at gain0.025. Louder audio does not improve model scores.
- MIDI-like uses a different generator and was not affected by the Basic final-Bar bug. Its unchanged verified sources/protocol were reused.

## Reproduce packaging

```bash
python3 baseline/experiments/align_task_b.py --output runs/task_b_aligned
python3 baseline/experiments/align_bonus.py --output runs/bonus_aligned_v2
python3 HW2_scripts/eval_bonus.py --prompt_dir POP909/prompts/bonus --gen_midi_dir runs/bonus_aligned_v2 --output_csv runs/bonus_aligned_v2/metrics.csv
python3 baseline/experiments/align_bonus.py --output runs/bonus_aligned_v2 --summarize-metrics
python3 baseline/experiments/prepare_upload.py --stage audio --gain 1 --soundfont /path/to/GeneralUser-GS.sf2 --fluidsynth /path/to/fluidsynth
python3 baseline/experiments/prepare_upload.py --stage package --soundfont /path/to/GeneralUser-GS.sf2
```

Completed outputs are not overwritten. These commands document original workflow order, not instructions to rerun already completed experiments. Use new --output/--work directories after changing sources. Packaging reads the current v7report and does not upload files or run Git.

## Final handoff

1. Retain all nine Task B pairs and mark preferred versions in BEST_VERSIONS.md, selections.json, and report.
2. Replace changed README/code/report/selection/verification files in Drive. MIDI, WAV, models, and SoundFont are unchanged by the English-documentation/report update.
3. Confirm the TA can download the folder. Submit the PDF to NTU COOL and include the cloud link in its assignment comment.
4. If editing packaged files later, refresh verification/file_manifest.json.

Git staging, commit, and push remain author-controlled. Changes here are local until committed/pushed or explicitly synchronized. See RUNS_LAYOUT.md for archived output locations.
