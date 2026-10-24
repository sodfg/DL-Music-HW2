# HW2 submission package

- [Google Drive: models, generated music, report, and source snapshot](https://drive.google.com/drive/folders/1NRQpT-gkLJNPZY7wvyQdwWZ_qVDI4K6x?usp=sharing)
- [GitHub: source repository](https://github.com/sodfg/DL-Music-HW2)

Task A selection: t12. Preferred Task B versions:001=t08,002=t10,003=t08. See BEST_VERSIONS.md and selections.json. All nine Task B versions are included.

## Contents

| Directory/file | Purpose |
|---|---|
| report/ | Current PDF and editable PPTX; submit the PDF separately to NTU COOL |
| generation_results/task_a_t12/ | 20unconditional32-bar pieces,20 MIDI+20 WAV |
| generation_results/task_b/ | 3prompts x3settings,9 MIDI+9 WAV,all submitted |
| generation_results/bonus/ | 10melody-plus-piano pieces,16annotated bars each,10 MIDI+10 WAV |
| models/ | Four author-trained inference checkpoints,without optimizer state |
| code/ | Project source,tests,documentation,and course evaluation entry points |
| metrics/ | Corrected Basic/optional tables and reevaluated Bonus scores |
| audio_assets/ | GeneralUser GS piano SoundFont and its license |
| verification/ | File hashes,audio checks,MIDI alignment,and model provenance |
| BEST_VERSIONS.md | Human-readable listening selections |
| selections.json | Machine-readable selections and sharing URLs |
| requirements.txt | Python dependencies; FluidSynth is installed separately |

Total:39 MIDI+39 WAV. WAVs use a common gain of GAIN_VALUE,44100Hz stereo PCM16,with zero clipping. SI evaluation inputs remain at gain0.025; louder listening WAVs do not replace SI inputs.
Task B exports align8prompt+24continuation bars with timing error below0.05ms. Bonus exports follow16annotated bars with variable beat counts,not a forced4/4meter.

The package excludes training/preprocessed data,course prompts,caches,virtual environments,Git history,and other researchers' neural checkpoints. The embedded checkpoint vocabulary is necessary inference configuration. Obtain course data separately for preparation,prompted generation,and evaluation.

## Install and run inference

Python3.10+. Install a PyTorch build appropriate for your CPU/CUDA environment,then requirements.txt. Original training used torch2.6.0+cu124 on an RTX A5000. A compatible existing shared environment can be reused. Pretrained checkpoints are included,so no retraining is needed. Run commands from this package root.

```bash
python3 -m pip install -r requirements.txt
python3 code/baseline/generate.py --checkpoint models/baseline_6.pt --output generated/task_a --count 20 --bars 32 --temperature 1.2 --top-k 32 --top-p .95 --seed 123 --generation-policy boundary-v2
python3 code/baseline/audio/render_audio.py --input generated/task_a --output generated/task_a_wav --soundfont audio_assets/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain GAIN_VALUE
```

Install FluidSynth separately or provide its executable path with --fluidsynth. Rendering needs the included SoundFont and its license.

Task B:replace COURSE_BASIC_B with the external course prompt folder. This example reproduces preferred prompt001,t08; use prompt002,t10 and prompt003,t08 for the other preferred versions.

```bash
python3 code/baseline/generate.py --checkpoint models/baseline_6.pt --prompt COURSE_BASIC_B/001/001.mid --prompt-beats COURSE_BASIC_B/001/beat_midi.txt --prompt-bars 8 --bars 32 --count 1 --temperature .8 --top-k 32 --top-p .95 --seed 123 --generation-policy boundary-v2 --output generated/basic_b/001/t08
python3 code/baseline/experiments/align_task_b.py --input-root generated/basic_b --prompts COURSE_BASIC_B --output generated/basic_b_aligned --prompt-id 001 --setting t08
python3 code/baseline/audio/render_audio.py --input generated/basic_b_aligned/001/t08 --output generated/basic_b_wav/001/t08 --soundfont audio_assets/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain GAIN_VALUE
```

All nine formal versions remain included; the preference labels do not remove the alternatives.

Bonus:replace COURSE_BONUS with the external course Bonus prompt folder. Inference reads only melody.mid and beat_midi.txt,never GT piano/chord answers.

```bash
python3 code/baseline/bonus/generate.py --checkpoint models/bonus_accompaniment.pt --prompts COURSE_BONUS --output generated/bonus
python3 code/baseline/experiments/align_bonus.py --input generated/bonus --prompts COURSE_BONUS --output generated/bonus_aligned
python3 code/baseline/audio/render_audio.py --input generated/bonus_aligned --output generated/bonus_wav --soundfont audio_assets/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain GAIN_VALUE
python3 code/HW2_scripts/eval_bonus.py --prompt_dir COURSE_BONUS --gen_midi_dir generation_results/bonus --output_csv new_bonus_metrics.csv
```

The three-layer checkpoint uses the Basic generator with models/depth_3.pt. MIDI-like uses code/baseline/representations/generate_midi_like.py with models/midi_like_6.pt.
Reproducing metrics may require separately obtaining course data and MusDr; see code/baseline/README.md. Inference was tested in the existing local environment; a fresh installation was not independently tested.

## Final handoff

Replace changed files in the shared Drive folder,verify that the TA can download them,and submit the PDF plus cloud link to NTU COOL. Both sharing URLs appear on the report cover. No automatic cloud replacement,Git commit,or push is performed by the packaging script.
If packaged files are edited later,refresh verification/file_manifest.json.
