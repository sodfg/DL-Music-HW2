"""Train from scratch, hold out complete songs, save resumable checkpoints."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from model import Config, MusicTransformer
from music import dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/baseline')
    p.add_argument('--output', default='runs/baseline')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume')
    p.add_argument('--validation-fraction', type=float, default=.05,
                   help='Set 0 for final training on all 896 songs; evaluation then probes training data')
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.context, a.eval_every, a.eval_batches) < 1:
        p.error('Counts must be positive')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    data = Path(a.data)
    vocab = json.loads((data/'tok2idx.json').read_text())
    songs = json.loads((data/'sequences.json').read_text())
    random.Random(a.seed).shuffle(songs)
    if not 0 <= a.validation_fraction < 1:
        p.error('validation-fraction must be in [0,1)')
    n_val = max(1, round(len(songs)*a.validation_fraction)) if a.validation_fraction else 0
    if len(songs) < 2:
        raise ValueError('Need at least two songs for a disjoint validation split')
    val_songs, train_songs = songs[:n_val], songs[n_val:]
    def convert(items):
        return [torch.tensor([vocab[t] for t in s['tokens']], dtype=torch.long) for s in items]
    training = convert(train_songs)
    validation = convert(val_songs) if val_songs else training
    eval_label = 'validation' if val_songs else 'training_probe'
    train_rng, val_rng = random.Random(a.seed+1), random.Random(a.seed+2)
    def batch(pool, rng):
        # Uniform song sampling, then a random crop. Never cross song boundaries.
        x = torch.full((a.batch_size, a.context), vocab['<PAD>'], dtype=torch.long)
        y = x.clone()
        for i in range(a.batch_size):
            seq = rng.choice(pool)
            start = rng.randint(0, max(0, len(seq)-a.context-1))
            chunk = seq[start:start+a.context+1]
            n = len(chunk)-1
            x[i,:n], y[i,:n] = chunk[:-1], chunk[1:]
        return x.to(a.device), y.to(a.device)
    config = Config(len(vocab), a.context, a.width, a.layers, a.heads)
    model = MusicTransformer(config).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    start, best = 0, float('inf')
    if a.resume:
        saved = torch.load(a.resume, map_location=a.device, weights_only=False)
        if saved['config'] != asdict(config) or saved['vocab'] != vocab:
            raise ValueError('Resume config/vocabulary mismatch')
        if saved['split'] != {'train': [s['song_name'] for s in train_songs], 'validation': [s['song_name'] for s in val_songs]}:
            raise ValueError('Resume dataset split mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best = saved['step'], saved['best_val']
        train_rng.setstate(saved['train_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    split = {'train': [s['song_name'] for s in train_songs], 'validation': [s['song_name'] for s in val_songs]}
    dump(output/'split.json', split)
    dump(output/'tok2idx.json', vocab)
    dump(output/'config.json', asdict(config))
    dump(output/'train_args.json', vars(a))
    mixed = a.device.startswith('cuda')
    def amp():
        return torch.autocast('cuda', dtype=torch.bfloat16) if mixed and torch.cuda.is_bf16_supported() else nullcontext()
    def save(name, step):
        destination = output/name
        temporary = output/(name+'.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': asdict(config), 'vocab': vocab, 'step': step, 'best_val': best,
                    'split': split, 'train_rng': train_rng.getstate(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if mixed else None}, temporary)
        temporary.replace(destination)
    print(f'Device={a.device}; parameters={sum(p.numel() for p in model.parameters()):,}; '
          f'train={len(training)}, held_out={len(val_songs)}, evaluation={eval_label}', flush=True)
    clock = time.time()
    for step in range(start+1, a.steps+1):
        model.train()
        lr = a.lr * min(1., step/200) * (.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        x,y = batch(training, train_rng)
        optimizer.zero_grad(set_to_none=True)
        with amp():
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1,len(vocab)), y.reshape(-1), ignore_index=vocab['<PAD>'])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(f'step={step} loss={loss.item():.4f} elapsed={time.time()-clock:.1f}s', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            val_rng.seed(a.seed+2)  # Repeat the same validation crops at every evaluation.
            total, count = 0., 0
            with torch.no_grad(), amp():
                for _ in range(a.eval_batches):
                    vx,vy = batch(validation, val_rng)
                    pred = model(vx)
                    total += F.cross_entropy(pred.reshape(-1,len(vocab)), vy.reshape(-1),
                                ignore_index=vocab['<PAD>'], reduction='sum').item()
                    count += (vy != vocab['<PAD>']).sum().item()
            score = total/count
            row = {'step': step, 'train_loss': loss.item(), eval_label+'_loss': score, 'lr': lr}
            with (output/'loss.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            improved = score < best
            best = min(best, score)
            save('last.pt', step)
            if improved:
                save('best.pt', step)
            print(f'{eval_label}={score:.4f}, best={best:.4f}; checkpoint saved', flush=True)

if __name__ == '__main__':
    main()
