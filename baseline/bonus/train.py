"""Train paired MELODY→PIANO from scratch; all songs, random bounded windows."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.bonus.data import training_pair,vocabulary
from baseline.bonus.model import Config,AccompanimentTransformer
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import save_json


def batch(songs,rng,vocab,size,target_context,device,seen=None):
    pairs=[]
    for _ in range(size):
        song=rng.choice(songs)
        if seen is not None:seen.add(song['song_name'])
        pairs.append(training_pair(song,rng,vocab,target_context))
    source=torch.zeros(size,max(len(s) for s,t in pairs),dtype=torch.long)
    length=max(len(t)-1 for s,t in pairs)
    target=torch.zeros(size,length,dtype=torch.long);labels=target.clone()
    for i,(s,t) in enumerate(pairs):
        source[i,:len(s)]=torch.tensor(s)
        target[i,:len(t)-1]=torch.tensor(t[:-1]);labels[i,:len(t)-1]=torch.tensor(t[1:])
    return source.to(device),target.to(device),labels.to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/bonus'))
    p.add_argument('--output',type=Path,default=Path('runs/bonus_accompaniment/train'))
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--target-context',type=int,default=1024)
    p.add_argument('--source-context',type=int,default=0,help='0 chooses max complete16bar melody length in data')
    p.add_argument('--width',type=int,default=384)
    p.add_argument('--encoder-layers',type=int,default=3)
    p.add_argument('--decoder-layers',type=int,default=4)
    p.add_argument('--heads',type=int,default=6)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--eval-batches',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',type=Path)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.target_context,a.width,a.encoder_layers,a.decoder_layers,a.heads,a.eval_every,a.eval_batches)<1 or a.width%a.heads:
        p.error('Invalid dimensions/counts')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:p.error('Training output exists; inspect and use --resume or a new output')
    random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    vocab=json.loads((a.data/'tok2idx.json').read_text())
    if vocab!=vocabulary():p.error('Bonus vocabulary mismatch')
    songs=json.loads((a.data/'pairs.json').read_text())
    summary=json.loads((a.data/'summary.json').read_text())
    if summary['songs']!=len(songs) or len({s['song_name'] for s in songs})!=len(songs):p.error('Dataset population mismatch')
    if summary['max_target_tokens_one_bar']>a.target_context:p.error('Target context cannot hold every full piano bar')
    source_context=a.source_context or max(256,math.ceil(summary['max_source_tokens_16_bars']/64)*64)
    if source_context<summary['max_source_tokens_16_bars']:p.error('Source context would truncate some melodies')
    config=Config(len(vocab),source_context,a.target_context,a.width,a.encoder_layers,a.decoder_layers,a.heads)
    identity={'pairs_sha256':sha256(a.data/'pairs.json'),'vocab_sha256':sha256(a.data/'tok2idx.json'),
              'songs':[s['song_name'] for s in songs],'config':asdict(config)}
    model=AccompanimentTransformer(config).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    rng=random.Random(a.seed+1);probe_rng=random.Random(a.seed+2);seen=set();start=0;best=float('inf')
    if a.resume:
        saved=torch.load(a.resume,map_location=a.device,weights_only=False)
        if saved['identity']!=identity or saved['train_steps']!=a.steps:p.error('Resume data/config/schedule differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['train_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if a.device.startswith('cuda') and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start=saved['step'];best=saved['best_probe'];seen=set(saved.get('seen_songs',[]))
    a.output.mkdir(parents=True,exist_ok=True)
    save_json(a.output/'identity.json',identity)
    save_json(a.output/'config.json',asdict(config))
    save_json(a.output/'train_args.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    mixed=a.device.startswith('cuda') and torch.cuda.is_bf16_supported()
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16) if mixed else nullcontext()
    def save(name,step):
        path=a.output/name;temporary=path.with_suffix('.pt.tmp')
        torch.save({'kind':'bonus_encoder_decoder','config':asdict(config),'vocab':vocab,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'step':step,'train_steps':a.steps,'best_probe':best,
            'identity':identity,'train_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None,'seen_songs':sorted(seen)},temporary)
        temporary.replace(path)
    print(f'Device={a.device}; encoder={a.encoder_layers}, decoder={a.decoder_layers}; parameters={sum(x.numel() for x in model.parameters()):,}; songs={len(songs)}; source_context={source_context}; evaluation=training_probe',flush=True)
    clock=time.monotonic()
    for step in range(start+1,a.steps+1):
        model.train();lr=a.lr*min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        source,target,labels=batch(songs,rng,vocab,a.batch_size,a.target_context,a.device,seen)
        optimizer.zero_grad(set_to_none=True)
        with amp():loss=F.cross_entropy(model(source,target).reshape(-1,len(vocab)),labels.reshape(-1),ignore_index=0)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1 or step%25==0:print(f'step={step} loss={loss.item():.4f} elapsed={time.monotonic()-clock:.1f}s',flush=True)
        if step%a.eval_every==0 or step==a.steps:
            model.eval();probe_rng.seed(a.seed+2);total=count=0
            with torch.no_grad(),amp():
                for _ in range(a.eval_batches):
                    src,tgt,y=batch(songs,probe_rng,vocab,a.batch_size,a.target_context,a.device)
                    pred=model(src,tgt)
                    total+=F.cross_entropy(pred.reshape(-1,len(vocab)),y.reshape(-1),ignore_index=0,reduction='sum').item()
                    count+=y.ne(0).sum().item()
            score=total/count;improved=score<best;best=min(best,score)
            with (a.output/'loss.jsonl').open('a') as stream:stream.write(json.dumps({'step':step,'loss':loss.item(),'training_probe':score,'lr':lr,'seen_songs':len(seen)})+'\n')
            save('last.pt',step)
            if improved:save('best.pt',step)
            print(f'training_probe={score:.4f}; best={best:.4f}; seen_songs={len(seen)}; checkpoint saved',flush=True)


if __name__=='__main__':main()
