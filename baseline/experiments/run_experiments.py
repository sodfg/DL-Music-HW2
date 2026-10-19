"""Run the three Basic A settings and all nine Basic B continuations."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',default='runs/experiments')
p.add_argument('--prompts',default='POP909/prompts/basic-b')
p.add_argument('--device',default='cuda')
p.add_argument('--count',type=int,default=20)
p.add_argument('--max-events-per-bar',type=int,default=128)
p.add_argument('--seed',type=int,default=123)
a=p.parse_args()
script=Path(__file__).resolve().parents[1]/'generate.py'
prompts=Path(a.prompts)
for name in ('001','002','003'):
    for f in (prompts/name/f'{name}.mid',prompts/name/'beat_midi.txt'):
        if not f.is_file():
            p.error(f'Missing prompt: {f}')
base=[sys.executable,str(script),'--checkpoint',a.checkpoint,'--device',a.device,
      '--seed',str(a.seed),'--max-events-per-bar',str(a.max_events_per_bar)]
for temperature in (.8,1.,1.2):
    label=f't{round(temperature*10):02d}'
    options=['--temperature',str(temperature),'--top-k','32','--top-p','.95']
    subprocess.run(base+options+['--count',str(a.count),'--bars','32','--output',
                                str(Path(a.output)/'basic_a'/label)],check=True)
    for name in ('001','002','003'):
        subprocess.run(base+options+['--count','1','--bars','32','--prompt-bars','8',
                       '--prompt',str(prompts/name/f'{name}.mid'),
                       '--prompt-beats',str(prompts/name/'beat_midi.txt'),
                       '--output',str(Path(a.output)/'basic_b'/name/label)],check=True)
