# Run-directory guide

The local [runs index](../../runs/00_README.md) labels each directory. runs/ is excluded from new Git additions by .gitignore; existing tracked dataset files are not automatically untracked by an ignore rule.

| Current source | Purpose |
|---|---|
| corrected_boundary_v2 | Corrected Basic/optional generation and formal summary |
| task_b_aligned | Nine final32-bar Task B MIDI files |
| bonus_aligned_v2 | Ten final16-annotated-bar Bonus MIDI files and reevaluated metrics |
| baseline | Six-layer checkpoint |
| depth_3_vs_6 | Three-layer checkpoint and reference/provenance records |
| representation_midi_like | MIDI-like checkpoint and verified samples/scores |
| bonus_accompaniment | Bonus checkpoint and original generation provenance |
| submission_state | Author-confirmed selections and sharing links |
| upload_preparation_v2 | Source of39 louder submission WAVs |
| report_build_v7 | Current report construction and validation |
| report_build/fonts.conf | Retained font configuration dependency |

_smoke holds execution pilots. _archive holds superseded plans, early samples, diagnostics, alignment trials, packaging trials, and report builders. Original file hashes and a relocation map are retained. Files inside archived snapshots may refer to their original locations; archival does not imply resumability from a new path.

experiments contains pre-fix generation; metrics mixes historical generated scores with reference inputs; audio_trial/listening retain old audio and legacy defaults; sampling_pilot is exploratory. These are not current submission sources. Do not blindly delete them: formal records can still reference historical inputs.

New small tests should explicitly use runs/_smoke/name. Formal experiments need descriptive fresh output names plus configuration/provenance. Upload only output/UPLOAD_HW2_B12901157, not all runs. This reorganization was local, did not delete original data, and did not synchronize EDA or run Git operations.
