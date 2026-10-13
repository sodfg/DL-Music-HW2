"""Render MIDI to WAV with FluidSynth and a user-supplied piano SoundFont."""
import argparse
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--input', required=True, help='MIDI file or directory')
p.add_argument('--output', required=True)
p.add_argument('--soundfont', required=True, help='Path to .sf2 SoundFont')
p.add_argument('--sample-rate', type=int, default=44100)
a = p.parse_args()
if not shutil.which('fluidsynth'):
    raise SystemExit('FluidSynth is required. See README; no audio has been rendered.')
if not Path(a.soundfont).is_file():
    raise SystemExit('SoundFont file does not exist')
source = Path(a.input)
files = sorted(list(source.glob('*.mid'))+list(source.glob('*.midi'))) if source.is_dir() else [source]
if not files:
    raise SystemExit('No MIDI files found')
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
for midi in files:
    target = out/(midi.stem+'.wav')
    subprocess.run(['fluidsynth','-ni',a.soundfont,str(midi),'-F',str(target),'-T','wav',
                    '-r',str(a.sample_rate)], check=True)
    print(target)
