# Usage guide

Start with the [project README](../README.md) for installation, data layout, training, and generation.
Run every command below from the repository root. Dataset files, checkpoints, audio, and results
are stored outside Git; a fresh clone does not contain them.

## Architecture

1. `music.py` converts MIDI and `beat_midi.txt` annotations into musical events and back.
2. `prepare.py` writes `data/baseline/{sequences,tok2idx,summary}.json` and `roundtrip.mid`.
3. `model.py` defines a randomly initialized causal Transformer: 6 layers, width 512, 8 heads,
   and a 1,024-token context. `train.py` learns to predict the next token.
4. `train.py` saves model weights, configuration, vocabulary, and training state in checkpoints
   under `runs/baseline/`. `best.pt` is selected by probe loss; `last.pt` supports resuming.
5. `generate.py` samples tokens under grammar constraints, then writes MIDI and generation records.
6. `evaluation/` computes metrics; `audio/` renders WAV; `experiments/` runs batch comparisons.

The custom representation uses `Bar`, `Beats_N`, `Pos_N`, `Tempo_N`, `Pitch_N`, `Vel_N`, and `Dur_N`.
A beat has 16 positions; velocity has 32 levels; durations span 1–256 positions. Tempo is quantized
to 5 BPM from 30–210. `Beats_N` records annotated beats per bar, not a complete time signature.
The three piano tracks are merged; pedal and detailed performance timing are not retained by
ordinary token decoding. Check the MIDI round trip before training. TA-generated JSON uses a
different vocabulary and cannot replace this pipeline's prepared data.

## Generate and continue

Run all three temperatures (0.8, 1.0, 1.2), each with top-k 32 and top-p 0.95:

```bash
python baseline/experiments/run_experiments.py \
  --checkpoint runs/baseline/best.pt --output runs/experiments_boundary_v2 --device cuda
```

This produces 60 Task A pieces (20 per setting) and 9 Task B pieces (3 prompts × 3 settings).
Use `--device cpu` without CUDA. The runner needs all three course prompts before it starts.
Task A sample numbers are output indices, not POP909 song IDs.

For a single Task B continuation:

```bash
python baseline/generate.py --checkpoint runs/baseline/best.pt \
  --prompt POP909/prompts/basic-b/001/001.mid \
  --prompt-beats POP909/prompts/basic-b/001/beat_midi.txt \
  --prompt-bars 8 --bars 32 --count 1 --temperature 1.0 \
  --output runs/my_continuation --generation-policy boundary-v2
```

`--bars 32` includes the 8-bar prompt. Output preserves the original prompt notes' timing,
pitch, and velocity; notes crossing the boundary are clipped. The continuation uses quantized
prompt tokens as context. Generation uses a sliding context without a KV cache and can take
longer than training. Generated bars contain 1–128 notes by default.

For the ending-pileup diagnostic, pass `--generation-policy boundary-v2` to
`generate.py` and use a NEW output directory. This prevents simultaneous duplicate
onset/pitch pairs while allowing chords and repeated notes at later times. In the
last requested bar, a sampled `Bar` is translated to `<EOS>` without adding another
bar; the model can finish at its learned bar boundary instead of relying solely on
the rarely learned end-of-song token. Original note-duration cropping remains.
The default is `legacy` to preserve historical experiment behavior. Results from
the two policies must be labeled separately and evaluated again; historical metric
tables do not apply to newly generated files. This changes inference, not weights.

The formal corrected batch and its report workflow are described in
[CORRECTED_GENERATION.md](docs/CORRECTED_GENERATION.md). New results live under
`runs/corrected_boundary_v2`; the older report and submission draft are historical
backups and must not be submitted as the corrected results.

## H4 / GS evaluation

Obtain MusDr once, then evaluate the prepared reference and each generated setting:

```bash
git clone https://github.com/slSeanWU/MusDr.git HW2_scripts/MusDr
python baseline/evaluation/evaluate.py --data data/baseline --output runs/metrics_boundary_v2/real.csv
for setting in t08 t10 t12; do
  python baseline/evaluation/evaluate.py \
    --data "runs/experiments_boundary_v2/basic_a/$setting" --output "runs/metrics_boundary_v2/$setting.csv"
done
python baseline/evaluation/summarize_metrics.py --input-dir runs/metrics_boundary_v2
```

Read `runs/metrics_boundary_v2/summary.md` for means, sample standard deviations, and absolute differences
from the real-data means. CSV files retain per-piece scores and aggregate values.
H4 measures pitch-class entropy over four bars; GS measures similarity of note-onset patterns.
The evaluator maps every bar to 64 relative positions for **both** real and generated music.
Do not mix these scores with results using a different grid. Compare metrics separately;
closeness to the reference mean does not establish originality or listening quality.

## WAV and optional experiments

Install FluidSynth separately and obtain a licensed `.sf2` SoundFont. Replace the example
SoundFont path with your actual file:

```bash
python baseline/audio/render_audio.py --input runs/experiments_boundary_v2/basic_a/t10 \
  --output runs/audio_boundary_v2/t10 --soundfont /path/to/piano.sf2 --gain 0.025
```

Use `--fluidsynth /path/to/fluidsynth` if the executable is not on PATH. Keep the same SoundFont,
gain, and rendering settings across audio comparisons. SI has separate dependencies in
`baseline/requirements-si.txt`; it is not needed for MIDI generation or H4/GS.

Detailed experiment notes are currently in Traditional Chinese:

| Experiment | Notes |
|---|---|
| WAV rendering and 48-second SI | [Optional workflow](docs/OPTIONAL_WORKFLOW.md) |
| 3-layer vs. 6-layer Transformer | [Depth experiment](docs/DEPTH_EXPERIMENT.md) |
| Bar/position vs. MIDI-like tokens | [Representation experiment](docs/REPRESENTATION_EXPERIMENT.md) |
| Sampling comparisons | [Sampling pilot](docs/SAMPLING_PILOT.md) |

These runners support planning before execution. Inspect `--help` and use a new output directory
for a changed experiment; do not overwrite historical plans or mix smoke results with formal runs.

## GPU, remote runs, and checks

Select a GPU explicitly with `CUDA_VISIBLE_DEVICES=0` before a Python command. Follow the host's
GPU-sharing rules. On EDA Union, keep the existing compatible shared PyTorch installation;
install only missing packages instead of replacing the shared environment.
For a long remote run, start `tmux new -s hw2`, execute the training command, then detach with
Ctrl+B followed by D. Reconnect with `tmux attach -t hw2`. tmux keeps the terminal session alive;
it does not select a GPU or automatically capture a process started elsewhere.

Resume with the same architecture, vocabulary, seed, and training settings:

```bash
python baseline/train.py --steps 20000 --validation-fraction 0 \
  --output runs/baseline --resume runs/baseline/last.pt
```

Changing total steps also changes the learning-rate schedule. The default training command
without `--validation-fraction 0` holds out approximately 5% of songs for validation.

A **smoke test** is a short execution check, not a quality evaluation:

```bash
python baseline/train.py --output runs/_smoke/quickcheck --device cpu --steps 2 \
  --context 32 --width 32 --layers 1 --heads 4 --batch-size 2 --eval-every 2 --eval-batches 1
python baseline/generate.py --checkpoint runs/_smoke/quickcheck/best.pt \
  --output runs/_smoke/quickcheck/generated --device cpu --count 1 --bars 2 --max-events-per-bar 4
python -m unittest discover -s baseline/tests -p 'test_*.py'
```

The baseline has been run locally on CPU and on an RTX A5000; H4/GS, WAV, SI, and depth comparisons
have also been executed. This does not imply every dependency version or optional configuration
has been tested. Keep code and dependency lists in Git; back up datasets and `runs/` separately.
