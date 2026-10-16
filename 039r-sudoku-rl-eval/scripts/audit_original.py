"""Secondary audit only: original 16 samples, never a new replication."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1]
source=root.parent/'039-sudoku-rl-gemma4/results/e4b_l4/results'
a,b=[np.array([np.mean(r['solved']) for r in json.loads((source/f'eval_{arm}.json').read_text())['rows']]) for arm in ['base','trained']]
rng=np.random.default_rng(39039)
d=np.concatenate([b[rng.integers(len(b),size=(1000,len(b)))].mean(1)-a[rng.integers(len(a),size=(1000,len(a)))].mean(1) for _ in range(100)])
out=dict(label='Original data, secondary audit, not a rerun',n_per_arm=len(a),puzzles=5,base=float(a.mean()),trained=float(b.mean()),difference=float(b.mean()-a.mean()),ci95=np.quantile(d,[.025,.975]).tolist(),decision_ci975=np.quantile(d,[.0125,.9875]).tolist())
(root/'results/original_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
