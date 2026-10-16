from pathlib import Path
import subprocess,json,time,importlib.metadata as m
info={'gpu':subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'),'time':time.time()}
info['versions']={}
for x in ['torch','unsloth','unsloth_zoo','transformers','trl','tokenizers']:
 try: info['versions'][x]=m.version(x)
 except: pass
Path('/content/results/provenance.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info))
