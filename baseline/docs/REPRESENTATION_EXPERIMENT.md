# Bar/Pos versus MIDI-like representation

Both models use the same896 quantized songs, six decoder layers, width512, heads8, dropout0.1, context1024, batch8,20000 steps, lr0.0003, seed42, and all-data training.

| Property | Bar/Pos | MIDI-like |
|---|---|---|
| Events | Bar/Beats/Tempo/Pos/Pitch/Vel/Dur | Tempo/Vel/NoteOn/NoteOff/TimeShift |
| Vocabulary | 1001 | 584 |
| Tokens including BOS/EOS | 6,258,677 | 6,258,395 |
| Note onsets | 1,508,536 | 1,508,536 |
| Model parameters | 19,952,128 | 19,738,624 |
| Approximate average beats per1024 tokens | 48.85 | 48.85 |

This is a custom beat-grid MIDI-like variant, not a claim of standard MidiTok compatibility. TimeShift units are1/16 annotated beat, not milliseconds; a16-step shift is one beat. Intervals longer than256 steps are split. Data are converted from baseline quantized sequences using beat annotations, not resegmented from unreliable embedded MIDI meter.

## Conversion and training

```bash
python3 baseline/representations/midi_like.py
python3 baseline/experiments/run_representation_experiment.py --stage plan
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_representation_experiment.py --stage all
```

Conversion verifies every note-on/off, velocity, tempo, overlap, boundary, and metric onset view. Same-pitch overlapping events are retained rather than overwritten in a pitch dictionary. FIFO duration pairing is used only in the metric view; H4/GS do not use duration. bar_metadata.json is used for validation, not as model input. Per-bar Tempo events still provide boundary cues.

Output: runs/representation_midi_like/train, generated, metrics.csv, and si48. Actual learned events are in event_sequences.json/event_tok2idx.json. sequences.json/tok2idx.json are a separate Bar/Pos onset view for the TA H4/GS evaluator; render audio from MIDI.

## Generation and evaluation

20 pieces,32 four-beat bars,T1/k32/p0.95/seeds123–142. A stateful external clock enforces length, per-bar tempo, at least one onset per bar, a128-onset cap, maximum256-step notes, valid NoteOn/Off transitions, and closing active notes at completion. Legal candidate sets differ from Bar/Pos, so this comparison includes inference constraints and context coverage as well as learned representation. Different vocabularies also change embedding size and cross-entropy scale. Equal seeds do not mean corresponding songs.

H4/GS use the same64-position bar normalization. SI uses matched SoundFont, gain0.025, and the first48 seconds. Existing results are reused only after source MIDI/WAV and complete algorithm/version/protocol checks. Short/mid/long SI bounds are3–8/8–15/15–48 seconds. Reference H4/GS uses896 full songs; SI uses20 sampled reference excerpts, so scopes are reported separately.

Current formal comparisons are in runs/corrected_boundary_v2/summary.md. Historical Bar/Pos results in the original representation report were superseded after the ending fix. MIDI-like was unaffected by that generator bug and its verified scores remain valid: H4=2.5307, GS=0.7885, SI=0.2973/0.1744/0.0247. One training seed cannot prove a universal representation ranking. This optional study covers Task A, not a replacement for Task B.
