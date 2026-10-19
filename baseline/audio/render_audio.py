"""Reproducible offline MIDI -> piano WAV rendering with FluidSynth.

No GPU, model inference or training is involved. Original MIDI files stay intact.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

import mido


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path):
    """Validate PCM16 and return duration/level information, without normalizing."""
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError(f'{path}: expected uncompressed PCM16 WAV')
        rate, channels, frames = stream.getframerate(), stream.getnchannels(), stream.getnframes()
        count, squared, peak, clipped = 0, 0, 0, 0
        while True:
            raw = stream.readframes(16384)
            if not raw:
                break
            samples = array('h')
            samples.frombytes(raw)
            if sys.byteorder != 'little':
                samples.byteswap()
            count += len(samples)
            squared += sum(int(x)*int(x) for x in samples)
            peak = max(peak, max(abs(x) for x in samples))
            clipped += sum(x in (-32768, 32767) for x in samples)
    if not count or peak == 0:
        raise ValueError(f'{path}: empty or silent audio')
    if count != frames*channels:
        raise ValueError(f'{path}: truncated PCM data')
    return {'sample_rate': rate, 'channels': channels, 'sample_width_bytes': 2,
            'duration_seconds': frames/rate, 'peak_amplitude': peak/32768,
            'rms_amplitude': math.sqrt(squared/count)/32768,
            'clipped_samples': clipped}


def piano_copy(source, destination):
    """Set every used melodic channel to GM piano; preserve timing and controls."""
    midi = mido.MidiFile(str(source))
    channels = set()
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.channel == 9:
                    raise ValueError(f'{source}: percussion channel found; use --instrument midi')
                channels.add(msg.channel)
            if msg.type == 'program_change':
                msg.program = 0
            elif msg.type == 'control_change' and msg.control in (0, 32):
                msg.value = 0
    setup = mido.MidiTrack()
    for channel in sorted(channels):
        setup.extend([mido.Message('control_change', channel=channel, control=0, value=0),
                      mido.Message('control_change', channel=channel, control=32, value=0),
                      mido.Message('program_change', channel=channel, program=0)])
    # Type 0 permits only one track; insert at its beginning instead.
    if midi.type == 0:
        midi.tracks[0][0:0] = setup
    elif midi.type == 1:
        midi.tracks.insert(0, setup)
    else:
        raise ValueError('Asynchronous type-2 MIDI is not supported')
    midi.save(str(destination))


def collect_files(source, recursive=False):
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in ('.mid', '.midi'):
            raise ValueError('Input file must be .mid or .midi')
        return source.parent, [source]
    if not source.is_dir():
        raise ValueError(f'Input does not exist: {source}')
    files = sorted(p for p in (source.rglob('*') if recursive else source.iterdir())
                   if p.is_file() and p.suffix.lower() in ('.mid', '.midi'))
    if not files:
        raise ValueError(f'No MIDI files in {source}')
    outputs = [p.relative_to(source).with_suffix('.wav') for p in files]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Input MIDI filenames would produce duplicate WAV paths')
    return source, files


def save_manifest(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='MIDI file or directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--soundfont', required=True, type=Path)
    parser.add_argument('--fluidsynth', default='fluidsynth', help='Executable name or absolute path')
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--gain', type=float, default=.1)
    parser.add_argument('--instrument', choices=['piano', 'midi'], default='piano')
    parser.add_argument('--recursive', action='store_true', help='Preserve relative subdirectory structure')
    parser.add_argument('--limit', type=int, help='Render only the first N files for a small trial')
    parser.add_argument('--timeout', type=float, default=600, help='Maximum seconds per render')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing WAVs even if settings differ')
    args = parser.parse_args()
    if args.timeout <= 0 or args.sample_rate < 8000 or not math.isfinite(args.gain) or not 0 < args.gain <= 5 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid sample rate, gain or limit')
    try:
        base, files = collect_files(args.input, args.recursive)
        if not args.soundfont.is_file():
            raise ValueError(f'SoundFont not found: {args.soundfont}')
        binary = shutil.which(args.fluidsynth)
        if not binary:
            raise ValueError('FluidSynth executable not found; see baseline/docs/OPTIONAL_WORKFLOW.md')
        if args.limit:
            files = files[:args.limit]
        version_result = subprocess.run([binary, '--version'], check=True, text=True, capture_output=True, timeout=15)
        version = (version_result.stdout+'\n'+version_result.stderr).strip()
        settings = {'renderer': 'FluidSynth', 'renderer_version': version,
                    'soundfont_sha256': sha256(args.soundfont), 'sample_rate': args.sample_rate,
                    'gain': args.gain, 'instrument': args.instrument, 'format': 'PCM16 stereo',
                    'reverb': False, 'chorus': False, 'polyphony': 256, 'cpu_cores': 1,
                    'user_config': 'empty', 'normalization': 'none'}
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output/'render_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema_version': 1, 'files': {}}
        # Each output has its own settings, so interrupted or partial reruns remain auditable.
        rendered, skipped = 0, 0
        with tempfile.TemporaryDirectory(prefix='hw2-render-') as temporary:
            config = Path(temporary)/'empty.conf'
            config.write_text('')
            for index, source in enumerate(files, 1):
                relative = source.relative_to(base).with_suffix('.wav')
                target = args.output/relative
                key = relative.as_posix()
                source_hash = sha256(source)
                record = manifest['files'].get(key)
                if target.exists() and not args.overwrite:
                    if (record and record['settings'] == settings and record['source_sha256'] == source_hash
                            and record['wav_sha256'] == sha256(target)):
                        skipped += 1
                        print(f'[{index}/{len(files)}] unchanged: {target}', flush=True)
                        continue
                    raise ValueError(f'{target}: exists with unverified/different settings; use a new output folder or --overwrite')
                target.parent.mkdir(parents=True, exist_ok=True)
                midi_input = source
                if args.instrument == 'piano':
                    midi_input = Path(temporary)/'piano.mid'
                    piano_copy(source, midi_input)
                staged = Path(temporary)/'audio.wav'
                command = [binary, '-ni', '-q', '-f', str(config), '-R', '0', '-C', '0',
                           '-g', str(args.gain), '-r', str(args.sample_rate), '-T', 'wav', '-O', 's16',
                           '-o', 'synth.polyphony=256', '-o', 'synth.cpu-cores=1',
                           '-F', str(staged), str(args.soundfont), str(midi_input)]
                result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=args.timeout)
                stats = inspect_wav(staged)
                if stats['sample_rate'] != args.sample_rate or stats['channels'] != 2:
                    raise ValueError('Renderer produced unexpected sample rate or channel count')
                midi_seconds = mido.MidiFile(str(midi_input)).length
                if stats['duration_seconds'] < midi_seconds-.05:
                    raise ValueError(f'{source}: audio shorter than MIDI timeline')
                # Stage beside destination for an atomic replace, even across filesystems.
                local_stage = target.with_suffix('.wav.tmp')
                shutil.copyfile(staged, local_stage)
                local_stage.replace(target)
                manifest['files'][key] = {'source_path': str(source.resolve()), 'source_sha256': source_hash,
                    'soundfont_path': str(args.soundfont.resolve()), 'settings': settings,
                    'midi_duration_seconds': midi_seconds, 'audio': stats,
                    'wav_sha256': sha256(target), 'renderer_log': (result.stdout+result.stderr).strip()}
                save_manifest(manifest_path, manifest)
                rendered += 1
                warning = ' WARNING: PCM clipping; lower gain for ALL comparison groups' if stats['clipped_samples'] else ''
                print(f'[{index}/{len(files)}] {target}: {stats["duration_seconds"]:.2f}s, '
                      f'peak={stats["peak_amplitude"]:.3f}{warning}', flush=True)
        print(f'Finished: rendered={rendered}, unchanged={skipped}; manifest={manifest_path}')
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, subprocess.TimeoutExpired, wave.Error) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else ''
        parser.exit(1, f'Error: {error}\n{detail or ""}\n')


if __name__ == '__main__':
    main()
