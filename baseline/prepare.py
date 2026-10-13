"""Prepare every training song; fail loudly instead of silently skipping data."""
import argparse
from pathlib import Path
from collections import Counter
from music import encode, vocabulary, dump, decode

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='POP909/training')
p.add_argument('--output', default='data/baseline')
p.add_argument('--expected-songs', type=int, default=896)
p.add_argument('--roundtrip-song', default='001')
a = p.parse_args()
folders = sorted(x for x in Path(a.dataset).iterdir() if x.is_dir())
if len(folders) != a.expected_songs:
    raise ValueError(f'Expected {a.expected_songs} songs, found {len(folders)}')
vocab, songs, beat_counts = vocabulary(), [], Counter()
for folder in folders:
    seq = encode(folder/f'{folder.name}.mid', folder/'beat_midi.txt')
    unknown = set(seq)-vocab.keys()
    if unknown:
        raise ValueError(f'{folder.name}: unknown tokens {unknown}')
    songs.append({'song_name': folder.name, 'tokens': seq})
    beat_counts.update(t for t in seq if t.startswith('Beats_'))
out = Path(a.output)
dump(out/'tok2idx.json', vocab)
dump(out/'sequences.json', songs)
dump(out/'summary.json', {'songs': len(songs), 'tokens': sum(len(s['tokens']) for s in songs),
                          'vocab_size': len(vocab), 'annotated_beats_per_bar': dict(beat_counts)})
selected = next(s for s in songs if s['song_name'] == a.roundtrip_song)
decode(selected['tokens'], out/'roundtrip.mid')
print(f'Prepared {len(songs)} songs, vocabulary={len(vocab)}; written to {out}')
print('Listen to roundtrip.mid before a full training run; timing/velocity are quantized.')
