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

## Submission and downloads

- [Google Drive: report, checkpoints, generated MIDI/WAV, and source snapshot](https://drive.google.com/drive/folders/1NRQpT-gkLJNPZY7wvyQdwWZ_qVDI4K6x?usp=sharing)
- [GitHub: source code and documentation](https://github.com/sodfg/DL-Music-HW2)
- [Start here](START_HERE.md) and [submission guide](baseline/docs/SUBMISSION_GUIDE.md)

Selected Task A configuration: **t12**. Preferred Task B versions: **001=t08, 002=t10, 003=t08**.
All nine Task B MIDI/WAV pairs remain in the submission. The current report is v7, with both links on its first page.
Sharing links were supplied by the author; download permissions have not been independently verified.

## Use a pretrained model first

No POP909 download is needed for unconditional Task A inference. Start with the
[data access and pretrained-model guide](baseline/docs/DATA_ACCESS.md#use-a-pretrained-model-without-the-dataset).
It explains where to save the cloud checkpoint and how to generate one new piece.

## Setup and training with course data

Python 3.10 or 3.12; an NVIDIA GPU is recommended for training.
Run all commands from the repository root.

```bash
git clone https://github.com/sodfg/DL-Music-HW2.git
cd DL-Music-HW2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r baseline/requirements.txt
```

The **course-provided, preprocessed POP909 split is not redistributed here**.
Exact retraining requires access to that course version; the original upstream
release is not a drop-in replacement. See [data access and reproducibility](baseline/docs/DATA_ACCESS.md)
before running the preparation/training commands below. Users without the course
files can still use the supplied pretrained checkpoint for Task A generation.

Users with course access should arrange their private copy as follows:

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
  --output runs/my_samples --count 2 --bars 32 --generation-policy boundary-v2
```

Open `runs/my_samples/*.mid` in a MIDI player. Training and generation automatically use CUDA
when available, otherwise CPU; add `--device cpu` to force CPU.
`--validation-fraction 0` trains on all 896 songs; its checkpoint-selection loss is a training probe,
not an independent validation score.

**Already have a model?** Skip preparation and training and follow the
[pretrained-model command](baseline/docs/DATA_ACCESS.md#use-a-pretrained-model-without-the-dataset),
which uses the exported `models/baseline_6.pt` rather than the training checkpoint path above.
A compatible checkpoint contains the model configuration and token vocabulary; Task A inference
needs no dataset. Download the inference checkpoints from the cloud submission folder linked above.

The corrected generation and evaluation workflow is documented in
[CORRECTED_GENERATION.md](baseline/docs/CORRECTED_GENERATION.md). Current formal
results use `runs/corrected_boundary_v2`. Historical Bar/Pos generated scores and
older report drafts are archived and superseded.

## Project layout

| Path | Purpose |
|---|---|
| `baseline/{music,prepare}.py` | MIDI/token conversion and dataset preparation |
| `baseline/{model,train,generate}.py` | Model architecture, training, and inference |
| `baseline/{audio,evaluation,experiments}/` | WAV rendering, metrics, and experiment runners |
| `baseline/{representations,tests,docs}/` | Alternative tokens, tests, and detailed notes |
| `HW2_scripts/` | Course reference scripts |
| `POP909/`, `data/`, `runs/` | Dataset, prepared tokens, and experiment outputs; excluded from the submission package |

The current source tree excludes the previously tracked course dataset. Local
`POP909/` files remain available to their owner. Older commits still contain the
previously tracked data: this cleanup does not rewrite history.

See the [usage guide](baseline/README.md) for continuation, evaluation, WAV export, and experiments.
The baseline uses custom `Bar / Beats / Pos / Tempo / Pitch / Vel / Dur` events and a 6-layer,
512-dimensional Transformer. Metrics compare musical properties, not originality or overall quality.

Based on course examples; metrics use [MusDr](https://github.com/slSeanWU/MusDr).
WAV export requires [FluidSynth](https://www.fluidsynth.org/) and a separately obtained SoundFont.
