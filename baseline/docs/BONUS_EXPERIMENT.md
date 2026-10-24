# Melody-conditioned piano accompaniment

The Bonus uses a separate encoder-decoder Transformer trained from random initialization on896 MELODY-to-PIANO pairs. BRIDGE is not added to the target. Inference reads only melody.mid and beat_midi.txt; GT piano.mid and chord_midi.txt are evaluator-only files, never generation conditions.

| Setting | Value |
|---|---|
| Architecture | bidirectional encoder3, causal decoder4, cross-attention |
| Width / heads / FFN / dropout | 384 / 6 / 1536 / 0.1, pre-LayerNorm |
| Tokens | Bar/Beats/Tempo/Pos/Pitch/Vel/Dur,16positions per beat |
| Vocabulary | 1021, including encoder Focus_0..15 and TargetBars_1..4 |
| Source / target context | 2560 / 1024 |
| Training | AdamW,lr0.0003,weight decay0.01,warmup200,cosine,clip1 |
| Steps / batch / seed | 20000 / 4 / 42 |
| Selection loss | training probe, not held-out validation |

Training examples retain up to16complete melody bars and up to4complete target bars. Targets too long for context are shortened to complete3/2/1-bar chunks rather than cut inside note events. Inference attends to all16melody bars and generates four independent4-bar accompaniment chunks with the appropriate Focus. Chunks do not condition on previous accompaniment, limiting cross-chunk continuity. Generation usesT1/k32/p0.95/seeds123–132,max48 onsets per bar,max256-step notes, and terminal clipping; empty bars are allowed but each chunk must contain a note.

## Workflow

```bash
python3 baseline/bonus/data.py
python3 baseline/experiments/run_bonus.py --stage plan
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_bonus.py --stage all
python3 -m unittest discover -s baseline/tests
```

Use the existing compatible EDA shared environment instead of replacing its PyTorch installation. The runner verifies checkpoint/source identities, completed training coverage of all896 songs, and complete generated groups. Interrupted compatible training may use --resume after inspection. Incompatible plans or partial generation require a new output directory.

Original model and generation provenance remain in runs/bonus_accompaniment. The writer retains melody tracks and creates a separate PIANO track/channel. Final submission files are in runs/bonus_aligned_v2. The beat file's final downbeat is a right boundary: there are16complete annotated bars, with varying beat counts. Do not assume all prompts are4/4. The final export maps each annotated beat to a display quarter and uses its bar beat count as meter numerator, without claiming to recover the original score denominator. Performance events are retained within0.027ms; the final beat can extend by less than1ms to preserve a note-off. Old end-of-track trailing silence can differ by a few milliseconds.

## Evaluation and submission

```bash
python3 HW2_scripts/eval_bonus.py --prompt_dir POP909/prompts/bonus \
  --gen_midi_dir runs/bonus_aligned_v2 --output_csv runs/bonus_aligned_v2/metrics.csv
python3 baseline/experiments/align_bonus.py --output runs/bonus_aligned_v2 --summarize-metrics
```

All ten aligned pieces were reevaluated with the unchanged TA evaluator:160 bars. Tiny retiming can affect duration/IOI histogram bins, so current results use DD0.7166 and DIOI0.5670 rather than the previous export's scores. CA0.1125,CA-ceiling0.7625,DP0.8549,DV0.6066.

CA compares estimated chords with supplied labels; the estimator supports24 major/minor triads and unsupported/no-chord labels affect attainable scores. CA-ceiling is a GT-estimator diagnostic, not a strict bound. OA compares per-bar pitch-class, velocity, duration, and inter-onset distributions. Pooled means are weighted by bars, with song sample standard deviations reported separately. High overlap does not guarantee good harmony or listening quality.

Submit ten melody-plus-piano MIDI files and ten WAVs; do not substitute GT accompaniment. Submission WAVs use gain1.0; SI evaluation audio elsewhere remains at gain0.025. See SUBMISSION_GUIDE.md.
