# Optional experiment workflow

1. Summarize H1/H4/GS with per-song means, sample standard deviations, finite-value counts, and absolute reference differences.
2. Render consistent piano WAVs and verify source hashes, SoundFont identity, duration, silence, and clipping.
3. Pilot SI costs before a matched48-second study.
4. Compare three- and six-layer models with the same Bar/Pos representation.
5. Compare Bar/Pos and MIDI-like using the six-layer architecture.

These studies are complete. Current formal tables are in runs/corrected_boundary_v2/summary.md. Historical output remains for provenance; do not mix pre-fix Bar/Pos scores into current tables. See DEPTH_EXPERIMENT.md, REPRESENTATION_EXPERIMENT.md, and CORRECTED_GENERATION.md.

## WAV rendering

```bash
python3 baseline/audio/render_audio.py \
  --input runs/corrected_boundary_v2/basic_a/t12 --output runs/_smoke/audio_preview \
  --soundfont /path/to/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain 1
```

FluidSynth renders existing notes without loading a neural model or needing a GPU. Use the same licensed SoundFont, piano bank0/program0,44100Hz stereo PCM16,polyphony256,single synthesis core,reverb/chorus off, and no per-song normalization. Temporary instrument changes do not modify source MIDI. Piano mode rejects percussion channels; --instrument midi retains instrument selection for other uses.

render_manifest.json records source/SoundFont/WAV hashes, renderer version/settings, duration, peak, RMS, and clipped sample count. Matching inputs/settings skip verified outputs; mismatches require a new directory or explicit --overwrite. Silent, clipped, truncated, or malformed outputs are rejected. WAVs may contain a release tail longer than the MIDI timeline. --recursive preserves subfolders, and --limit1 supports a pilot.

Formal SI uses gain0.025, while submission listening WAVs use gain1.0. Do not overwrite SI inputs with louder submission files. The same SoundFont in a MIDI player does not ensure identical audio because engines/effects differ. Preserve its license and hash. Local bundled runtimes are not portable system installations.

## SI

Use an isolated compatible SI environment from requirements-si.txt. Do not replace a working EDA training environment. SI runtime grows rapidly with excerpt length; a short pilot cannot be linearly extrapolated to full songs.

```bash
python3 baseline/experiments/run_si_study.py
python3 baseline/experiments/run_si_study.py --help
python3 baseline/evaluation/evaluate_si.py --help
```

Formal generated groups contain20complete songs. Reference excerpts use seed42 and20 songs sampled from895 length-eligible songs, analyzed over the first48 seconds. Bounds: short3–8s,mid8–15s,long15–48s. Use matching renderer, versions, algorithm/wrapper hash, SoundFont, gain, source identity, and excerpt start/duration before reusing a cache. Do not exclude short songs or change seeds to obtain compatible lengths.

SI describes repeated structure and can reward undesirable repetition. It is not plagiarism detection or a universal quality score. Reference H4/GS uses896 full songs; SI uses20excerpts, so keep scopes separate. Sample standard deviation describes between-song variation, not confidence intervals. A single training seed cannot establish significance.

Each independently working, verified feature can have a separate commit. Branches are optional. Git staging/commit/push remain author-controlled. New pilots belong under runs/_smoke; archive historical snapshots without rewriting their original plan hashes.
