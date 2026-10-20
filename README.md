# Piano Music Generation

Train a decoder-only Transformer from scratch on the course POP909 split and generate piano MIDI.

- **Task A:** generate 32 bars without a melody prompt.
- **Task B:** keep an 8-bar prompt and generate a 24-bar continuation.
- **Evaluation:** H4 and GS, with optional SI, model-depth, and token-representation experiments.

```text
MIDI + beat annotations → tokens → Transformer → generated tokens → MIDI
                                                                  ├─ WAV
                                                                  └─ metrics
```

## Quick start

Python 3.10 or 3.12; an NVIDIA GPU is recommended for training.
Run all commands from the repository root.

```bash
git clone https://github.com/sodfg/DL-Music-HW2.git
cd DL-Music-HW2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r baseline/requirements.txt
```

Obtain the **course-provided POP909 split** separately; the upstream dataset has a different layout:

```text
POP909/
  training/001/001.mid        # 896 song folders; each includes beat_midi.txt
  prompts/basic-b/001/001.mid # prompts 001–003; each includes beat_midi.txt
```

```bash
python baseline/prepare.py
# Check data/baseline/roundtrip.mid before training.
python baseline/train.py --steps 20000 --validation-fraction 0 --output runs/baseline
python baseline/generate.py --checkpoint runs/baseline/best.pt \
  --output runs/my_samples --count 2 --bars 32
```

Open `runs/my_samples/*.mid` in a MIDI player. Training and generation automatically use CUDA
when available, otherwise CPU; add `--device cpu` to force CPU.
`--validation-fraction 0` trains on all 896 songs; its checkpoint-selection loss is a training probe,
not an independent validation score.

**Already have a model?** Skip preparation and training and use the generation command above.
A compatible checkpoint contains the model configuration and token vocabulary; Task A inference
needs no dataset. No public checkpoint download is currently provided.

## Project layout

| Path | Purpose |
|---|---|
| `baseline/{music,prepare}.py` | MIDI/token conversion and dataset preparation |
| `baseline/{model,train,generate}.py` | Model architecture, training, and inference |
| `baseline/{audio,evaluation,experiments}/` | WAV rendering, metrics, and experiment runners |
| `baseline/{representations,tests,docs}/` | Alternative tokens, tests, and detailed notes |
| `HW2_scripts/` | Course reference scripts |
| `POP909/`, `data/`, `runs/` | Dataset, prepared tokens, and generated artifacts; excluded from Git |

See the [usage guide](baseline/README.md) for continuation, evaluation, WAV export, and experiments.
The baseline uses custom `Bar / Beats / Pos / Tempo / Pitch / Vel / Dur` events and a 6-layer,
512-dimensional Transformer. Metrics compare musical properties, not originality or overall quality.

Based on course examples; metrics use [MusDr](https://github.com/slSeanWU/MusDr).
WAV export requires [FluidSynth](https://www.fluidsynth.org/) and a separately obtained SoundFont.
