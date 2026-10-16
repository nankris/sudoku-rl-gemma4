import json
from pathlib import Path
p=Path(__file__).parent
n=json.loads((p/'original_recipe.ipynb').read_text())
def get(i): return ''.join(n['cells'][i]['source'])
def put(i,s): n['cells'][i]['source']=s
put(0,get(0).replace('MAX_STEPS = 100','MAX_STEPS = 24').replace('BETA = None','BETA = 0.001').replace('TRAIN_MINUTES = 50.0','TRAIN_MINUTES = 125.0'))
put(15,get(15).replace('EVAL_SAMPLES = 16','EVAL_SAMPLES = 64'))
put(18,get(18).replace('gradient_accumulation_steps = 2','gradient_accumulation_steps = 4').replace('num_generations = 2','num_generations = 4'))
put(22,get(22)+'\nimport shutil\nshutil.make_archive("/content/results/adapter", "zip", "/content/gemma_4_lora")')
put(23,get(23)+'''
import numpy as np

def compare(base_tags, trained_tags):
    def values(tags):
        rows = sum([json.load(open(f'/content/results/eval_{t}.json'))['rows'] for t in tags], [])
        return np.array([np.mean(r['solved']) for r in rows])
    a,b=values(base_tags),values(trained_tags)
    rng=np.random.default_rng(39039)
    diffs=np.concatenate([b[rng.integers(len(b),size=(1000,len(b)))].mean(1)-a[rng.integers(len(a),size=(1000,len(a)))].mean(1) for _ in range(100)])
    ci=np.quantile(diffs,[.025,.975]).tolist()
    decision_ci=np.quantile(diffs,[.0125,.9875]).tolist()
    verdict='PROVEN' if decision_ci[0]>0 else 'DISPROVEN' if decision_ci[1]<0 else 'STILL INCONCLUSIVE'
    return dict(n_per_arm=len(a),base=float(a.mean()),trained=float(b.mean()),difference=float(b.mean()-a.mean()),ci95=ci,decision_ci975=decision_ci,verdict=verdict)
r1=compare(['base'],['trained'])
json.dump(r1,open('/content/results/round1.json','w'),indent=2)
plog('ROUND1 '+json.dumps(r1))
if r1['verdict']=='STILL INCONCLUSIVE':
    with model.disable_adapter():
        run_eval('base_extra', seed=5678)
    run_eval('trained_extra', seed=6789)
    r2=compare(['base','base_extra'],['trained','trained_extra'])
    json.dump(r2,open('/content/results/round2.json','w'),indent=2)
    plog('ROUND2 '+json.dumps(r2))
plog('EXPERIMENT FINISHED')
''')
# Independent sampling streams per arm rather than nominal paired seeds.
put(23,get(23).replace("run_eval('trained')","run_eval('trained', seed=2345)"))
for c in n['cells']:
 c['outputs']=[];c['execution_count']=None
(p/'run.ipynb').write_text(json.dumps(n,indent=1))
