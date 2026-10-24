# Data access and reproducibility

[Start here](../../START_HERE.md) → **Data access** → [Usage guide](../README.md)

## What this repository provides

This project uses a course-provided, preprocessed POP909 split. The course dataset,
prompt MIDI files, and beat/chord annotations are not redistributed in the current
repository tree or submission package. They may be retained in a private local copy.

The [cloud submission](https://drive.google.com/drive/folders/1NRQpT-gkLJNPZY7wvyQdwWZ_qVDI4K6x?usp=sharing)
contains pretrained inference checkpoints, generated MIDI/WAV results, metrics, and
source code. The full report is submitted privately through NTU COOL. The public
folder is not a dataset download location.

| Goal | Additional data required? |
|---|---|
| Generate new unconditional Task A music using the supplied checkpoint | No dataset required |
| Continue a Task B prompt | A compatible prompt MIDI and its beat_midi.txt |
| Generate Bonus accompaniment | Compatible melody.mid and beat_midi.txt prompts |
| Retrain the reported models or reproduce all reference metrics | The exact course-provided split and annotations |

Exact retraining requires access to the course version. We do not provide a public
download for it. If you have course access, obtain the original course files through
the course materials. Otherwise, exact retraining and reproduction of the reported
reference scores are not guaranteed.

The [original POP909 project](https://github.com/music-x-lab/POP909-Dataset) is a
separate upstream resource, not a drop-in replacement for the course version.
Adapting it would require checking layout, preprocessing, annotations, and split;
results would need to be labeled as a different setup rather than the reported run.
No automatic dataset download is performed by this repository.

## Use a pretrained model without the dataset

From a clone of this repository, download `models/baseline_6.pt` from the cloud
submission and save it at `models/baseline_6.pt` in the clone. The checkpoint
contains the model configuration and vocabulary. After installing
`baseline/requirements.txt`, generate one new 32-bar piece:

```bash
python baseline/generate.py --checkpoint models/baseline_6.pt \
  --output generated/task_a_preview --count 1 --bars 32 \
  --temperature 1.2 --top-k 32 --top-p 0.95 --seed 123 \
  --generation-policy boundary-v2
```

Choose a new writable `--output` directory. Add `--device cpu` to force CPU
inference; CUDA is selected automatically when available. This command creates
MIDI; see the [WAV rendering instructions](../README.md#wav-and-optional-experiments)
to synthesize it with FluidSynth. The downloaded cloud package has a different
`code/baseline/` source prefix; follow its root README when running inside that
package rather than mixing the two directory layouts.

## Local course-data layout

Only users with the course files should run preparation, training, or full
reference evaluation. Place their private local copy at:

```text
POP909/
  training/
    001/
      001.mid
      beat_midi.txt
      chord_midi.txt
    ...                       # 896 training song folders
  prompts/
    basic-b/
      001/                    # prompts 001–003
        001.mid
        beat_midi.txt
        chord_midi.txt
    bonus/
      001/                    # prompts 001–010
        melody.mid
        piano.mid             # ground truth, evaluation only
        beat_midi.txt
        chord_midi.txt         # evaluation only
```

Basic preparation merges MELODY, BRIDGE, and PIANO. Bonus training pairs MELODY
with PIANO; Bonus inference never reads the ground-truth piano or chord files.
Training includes all 896 course training songs and uses a training probe rather
than an independent validation set in the reported configuration.

## Git tracking and private backups

`POP909/` and prepared `data/` are ignored. The dataset was first untracked without
deleting local files, then removed from the publication branch's history before
publication. The published `main` branch contains no course-data paths in its
current tree or history. The author retains the original Git history in a private
local recovery backup; that backup is not published.

Keep any desired dataset backup privately. The project does not promise permanent
hosting of the course dataset. Its current inference checkpoints are supplied
separately through the cloud submission.
