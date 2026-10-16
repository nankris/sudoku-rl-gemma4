import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'results'
rounds=[(i,json.loads((r/f'round{i}.json').read_text())) for i in [1,2] if (r/f'round{i}.json').exists()]
if rounds:
 fig,axs=plt.subplots(1,2,figsize=(9,4))
 for i,d in rounds:
  axs[0].bar([f'R{i} base',f'R{i} GRPO'],[100*d['base'],100*d['trained']],color=['#7d8da6','#008578'])
  axs[1].errorbar(100*d['difference'],i,xerr=[[100*(d['difference']-d['ci95'][0])],[100*(d['ci95'][1]-d['difference'])]],fmt='o',capsize=5,color='#008578')
 axs[0].set(ylabel='Puzzle solve rate (%)',ylim=(0,100))
 axs[1].axvline(0,color='gray',ls='--');axs[1].set(xlabel='GRPO minus base (percentage points), 95% CI',yticks=[i for i,d in rounds],yticklabels=[f'Round {i}' for i,d in rounds])
 fig.suptitle('039r: independent strategies; bootstrap clusters all five puzzles')
 fig.tight_layout();fig.savefig(ROOT/'chart.png',dpi=180)
 print(json.dumps(rounds,indent=2))
else:
 fig,ax=plt.subplots(figsize=(7,3));ax.axis('off');ax.text(.5,.5,'039r: no completed evaluation\nNo effect estimate available',ha='center',va='center',fontsize=16);fig.savefig(ROOT/'chart.png',dpi=180)
