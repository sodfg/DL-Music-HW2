"""Evaluate H1/H4/GS via TA MusDr code using a common normalized bar grid.

GS divides by the position-grid size. The full tokenizer reserves 512 positions,
so passing its complete dictionary would inflate scores for ordinary 4-beat bars.
Map every bar to 64 relative positions for both real and generated samples.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from music import dump, PPB


def normalize(songs):
    result=[]
    for song in songs:
        seq= song['tokens']
        normalized=[]
        beats=4
        for t in seq:
            if t == 'Bar':
                beats=4
                normalized.append(t)
            elif t.startswith('Beats_'):
                beats=int(t.split('_')[1])
            elif t.startswith('Pos_'):
                pos=int(t.split('_')[1])
                normalized.append(f'Pos_{min(63,pos*64//(beats*PPB))}')
            else:
                normalized.append(t)
        result.append({'song_name':song['song_name'],'tokens':normalized})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,help='Folder containing sequences.json and tok2idx.json')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    songs=json.loads((Path(a.data)/'sequences.json').read_text())
    old=json.loads((Path(a.data)/'tok2idx.json').read_text())
    keys=[t for t in old if not t.startswith(('Pos_','Beats_'))]+[f'Pos_{i}' for i in range(64)]
    vocab={t:i for i,t in enumerate(keys)}
    normalized=normalize(songs)
    if any(t not in vocab for s in normalized for t in s['tokens']):
        raise ValueError('Normalized evaluation contains unknown tokens')
    cache=output.parent/(output.stem+'_inputs')
    dump(cache/'tok2idx.json',vocab)
    dump(cache/'sequences.json',normalized)
    script=Path(__file__).resolve().parents[1]/'HW2_scripts/eval_basic.py'
    subprocess.run([sys.executable,str(script),'--tok2idx_path',str(cache/'tok2idx.json'),
                    '--sequences_path',str(cache/'sequences.json'),'--output_csv',str(output)],check=True)

if __name__ == '__main__':
    main()
