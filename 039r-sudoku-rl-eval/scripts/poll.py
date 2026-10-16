import subprocess,json,time
from pathlib import Path
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'))
for f in ['progress.log','run.log','round1.json','round2.json']:
 p=Path('/content/results')/f
 if p.exists(): print(f, p.read_text()[-3500:])
print('elapsed_seconds',time.time()-float(Path('/content/results/start_epoch.txt').read_text()))
