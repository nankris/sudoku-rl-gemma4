import subprocess,time
from pathlib import Path
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'))
print(subprocess.getoutput('ps -eo pid,ppid,etime,args | rg "nbconvert|ipykernel"') if False else subprocess.getoutput("ps -eo pid,ppid,etime,args | grep -E 'nbconvert|ipykernel' | head -10"))
p=Path('/content/results/progress.log');print(p.read_text()[-2000:] if p.exists() else 'no progress')
print(Path('/content/results/run.log').read_text()[-3000:])
