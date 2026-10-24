"""Generate exact-length MIDI-like pieces, with stateful valid-event constraints.

Keep baseline sampling settings, one onset per generated bar, and max onset cap.
Generation uses 4-beat bars; no Bar/Pos tokens are in the learned vocabulary.
"""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.model import Config, MusicTransformer
from baseline.music import PPB, MAX_DURATION, dump, vocabulary as bar_vocabulary
from baseline.representations.midi_like import vocabulary, decode, metric_tokens


class EventState:
    def __init__(self, bars=32, beats=4, max_events=128):
        self.bars, self.beats, self.max_events = bars, beats, max_events
        self.bar_steps = beats*PPB
        self.total = bars*self.bar_steps
        self.clock, self.onsets = 0, 0
        self.need_tempo, self.pending = True, False
        self.active = defaultdict(deque)
        self.vocab = vocabulary()

    def choices(self):
        if self.clock == self.total:
            offs = [f'NoteOff_{p}' for p in sorted(self.active) if self.active[p]]
            return offs or ['<EOS>']
        if self.need_tempo:
            return [t for t in self.vocab if t.startswith('Tempo_')]
        if self.pending:
            return [f'NoteOn_{p}' for p in range(128)]
        due = [f'NoteOff_{p}' for p in sorted(self.active)
               if self.active[p] and self.clock-self.active[p][0] >= MAX_DURATION]
        if due:
            return due
        result = [f'NoteOff_{p}' for p in sorted(self.active)
                  if self.active[p] and self.active[p][0] < self.clock]
        if self.onsets < self.max_events:
            result += [t for t in self.vocab if t.startswith('Vel_')]
        boundary = (self.clock//self.bar_steps+1)*self.bar_steps
        stop = boundary if self.onsets else boundary-1
        deadlines = [q[0]+MAX_DURATION for q in self.active.values() if q]
        if deadlines:
            stop = min(stop, min(deadlines))
        result += [f'TimeShift_{n}' for n in range(1, min(256, stop-self.clock)+1)]
        if not result:
            raise ValueError('No valid event continuation')
        return result

    def advance(self, token):
        if token not in self.choices():
            raise ValueError(f'Invalid transition at {self.clock}: {token}')
        if token == '<EOS>':
            return
        kind, value = token.split('_')
        value = int(value)
        if kind == 'Tempo':
            self.need_tempo = False
        elif kind == 'Vel':
            self.pending = True
        elif kind == 'NoteOn':
            self.active[value].append(self.clock)
            self.onsets += 1
            self.pending = False
        elif kind == 'NoteOff':
            self.active[value].popleft()
        else:
            self.clock += value
            if self.clock < self.total and self.clock % self.bar_steps == 0:
                self.onsets = 0
                self.need_tempo = True


def sample(logits, ids, temperature, top_k, top_p):
    # Same temperature/top-k/nucleus procedure as baseline/generate.py.
    scores = logits[ids].float()/temperature
    if top_k:
        scores[scores < torch.topk(scores, min(top_k,len(ids))).values[-1]] = -float('inf')
    if top_p < 1:
        sorted_scores, order = scores.sort(descending=True)
        remove = sorted_scores.softmax(0).cumsum(0) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[order[remove]] = -float('inf')
    return ids[torch.multinomial(scores.softmax(0), 1).item()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--bars', type=int, default=32)
    p.add_argument('--beats', type=int, default=4)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--top-k', type=int, default=32)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--max-events-per-bar', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    import math
    if (min(a.count,a.bars,a.beats,a.max_events_per_bar) < 1 or a.beats > 32
            or not math.isfinite(a.temperature) or a.temperature <= 0
            or not 0 < a.top_p <= 1 or a.top_k < 0):
        p.error('Invalid sampling settings')
    if (a.output/'event_sequences.json').exists() or list(a.output.glob('*.mid')):
        p.error('Output already contains samples; choose a new output directory')
    if a.device == 'cpu':
        torch.set_num_threads(min(4,torch.get_num_threads()))
    saved = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    vocab = vocabulary()
    if saved['vocab'] != vocab:
        p.error('Checkpoint does not use this MIDI-like vocabulary')
    model = MusicTransformer(Config(**saved['config'])).to(a.device)
    model.load_state_dict(saved['model'])
    model.eval()
    inverse = {i:t for t,i in vocab.items()}
    args = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    dump(a.output/'generation_args.json', args)
    dump(a.output/'event_tok2idx.json', vocab)
    dump(a.output/'tok2idx.json', bar_vocabulary())
    raw, views = [], []
    with torch.inference_mode():
        for index in range(a.count):
            torch.manual_seed(a.seed+index)
            state = EventState(a.bars, a.beats, a.max_events_per_bar)
            tokens = ['<BOS>']
            limit = a.bars*(4*a.max_events_per_bar+a.beats*PPB+4)+512
            while tokens[-1] != '<EOS>':
                if len(tokens)>limit:
                    raise RuntimeError('Generation guard exceeded')
                ids = torch.tensor([[vocab[t] for t in tokens[-model.config.context:]]], device=a.device)
                logits = model(ids)[0,-1]
                chosen = sample(logits,[vocab[t] for t in state.choices()],a.temperature,a.top_k,a.top_p)
                token = inverse[chosen]
                state.advance(token)
                tokens.append(token)
            name = f'sample_{index:03d}'
            decode(tokens, a.output/(name+'.mid'))
            raw.append({'song_name':name,'tokens':tokens})
            views.append({'song_name':name,'tokens':metric_tokens(tokens,[a.beats]*a.bars)})
            print(f'{name}: {a.bars} bars, {len(tokens)} event tokens, {state.clock} grid steps', flush=True)
    dump(a.output/'sequences.json', views)
    dump(a.output/'event_sequences.json', raw)


if __name__ == '__main__':
    main()
