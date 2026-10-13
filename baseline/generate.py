"""Grammar-constrained sampling, exact bar count, optional prompt continuation."""
import argparse
import json
from pathlib import Path
import torch
import pretty_midi
from model import Config, MusicTransformer
from music import encode, decode, dump, PPB, beat_grid, preserve_prompt


def allowed(tokens, vocab, target, generated_beats, max_events):
    last = tokens[-1]
    bars = tokens.count('Bar')
    bar_idx = len(tokens)-1-tokens[::-1].index('Bar') if bars else 0
    current = tokens[bar_idx:]
    groups = lambda prefix: [t for t in vocab if t.startswith(prefix+'_')]
    if last == '<BOS>':
        return ['Bar']
    if last == 'Bar':
        return [f'Beats_{generated_beats}']
    if last.startswith('Beats_'):
        return groups('Tempo')
    if last.startswith('Pitch_'):
        return groups('Vel')
    if last.startswith('Vel_'):
        return groups('Dur')
    beats = next(int(t.split('_')[1]) for t in current if t.startswith('Beats_'))
    pos = next((int(t.split('_')[1]) for t in reversed(current) if t.startswith('Pos_')), 0)
    if last.startswith('Pos_'):
        # An onset must be followed by one complete note.
        return groups('Pitch')
    ending = ['Bar'] if bars < target else ['<EOS>']
    if sum(t.startswith('Pitch_') for t in current) >= max_events:
        return ending
    choices = [f'Pos_{i}' for i in range(pos, beats*PPB)]
    # Require at least one note in every generated bar.
    return choices + (ending if any(t.startswith('Dur_') for t in current) else [])


def sample(logits, ids, temperature, top_k, top_p):
    scores = logits[ids].float()/temperature
    if top_k:
        scores[scores < torch.topk(scores, min(top_k,len(ids))).values[-1]] = -float('inf')
    if top_p < 1:
        sorted_scores, order = scores.sort(descending=True)
        cumulative = sorted_scores.softmax(0).cumsum(0)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[order[remove]] = -float('inf')
    return ids[torch.multinomial(scores.softmax(0), 1).item()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--bars', type=int, default=32)
    p.add_argument('--beats', type=int, default=4)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--top-k', type=int, default=32)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--max-events-per-bar', type=int, default=128)
    p.add_argument('--prompt', help='Prompt MIDI path; requires its beat_midi.txt')
    p.add_argument('--prompt-beats')
    p.add_argument('--prompt-bars', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.temperature <= 0 or not 0 < a.top_p <= 1 or a.top_k < 0:
        p.error('Invalid sampling parameters')
    if a.count < 1 or a.bars < 1 or not 1 <= a.beats <= 32 or a.max_events_per_bar < 1:
        p.error('Invalid piece dimensions')
    if a.prompt and (not a.prompt_beats or not 0 < a.prompt_bars < a.bars):
        p.error('Prompt requires --prompt-beats and 0 < prompt-bars < bars')
    if a.device == 'cpu':
        torch.set_num_threads(min(4,torch.get_num_threads()))
    saved = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    vocab = saved['vocab']
    inverse = {i:t for t,i in vocab.items()}
    model = MusicTransformer(Config(**saved['config'])).to(a.device)
    model.load_state_dict(saved['model'])
    model.eval()
    prefix = ['<BOS>']
    source = None
    if a.prompt:
        full = encode(a.prompt, a.prompt_beats)
        bars = [i for i,t in enumerate(full) if t == 'Bar']
        if len(bars) < a.prompt_bars:
            raise ValueError('Prompt has fewer annotated bars than requested')
        prefix = full[:bars[a.prompt_bars]] if len(bars) > a.prompt_bars else full[:-1]
        source = pretty_midi.PrettyMIDI(a.prompt)
        times, starts, sizes = beat_grid(a.prompt_beats)
        end_idx = starts[a.prompt_bars] if len(starts)>a.prompt_bars else starts[-1]+sizes[-1]
        original_boundary = float(times[end_idx])
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    dump(output/'tok2idx.json', vocab)
    dump(output/'generation_args.json', vars(a))
    pieces = []
    with torch.inference_mode():
        for n in range(a.count):
            torch.manual_seed(a.seed+n)
            tokens = prefix.copy()
            # Prefix stays untouched; the next generated token begins bar 9.
            if a.prompt:
                tokens.append('Bar')
            limit = a.bars*(4*a.max_events_per_bar+4)+len(prefix)+8
            while tokens[-1] != '<EOS>':
                if len(tokens)>limit:
                    raise RuntimeError('Generation guard reached; no partial result written')
                ids = torch.tensor([[vocab[t] for t in tokens[-model.config.context:]]], device=a.device)
                logits = model(ids)[0,-1]
                choices = allowed(tokens,vocab,a.bars,a.beats,a.max_events_per_bar)
                next_id = sample(logits,[vocab[t] for t in choices],a.temperature,a.top_k,a.top_p)
                tokens.append(inverse[next_id])
            name = f'{Path(a.prompt).stem + "_" if a.prompt else "sample_"}{n:03d}'
            path = output/f'{name}.mid'
            ticks = decode(tokens,path)
            if source:
                preserve_prompt(a.prompt, path, original_boundary, ticks[a.prompt_bars])
            pieces.append({'song_name':name,'tokens':tokens})
            print(f'{path}: {tokens.count("Bar")} bars, {len(tokens)} tokens', flush=True)
    dump(output/'sequences.json',pieces)

if __name__ == '__main__':
    main()
