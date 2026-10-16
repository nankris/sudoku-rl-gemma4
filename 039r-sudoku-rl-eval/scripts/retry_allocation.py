import subprocess,time,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
log=root/'results/allocation.jsonl'
for attempt in range(1,7):
 time.sleep(300)
 usage=subprocess.run(['colab','usage'],capture_output=True,text=True).stdout
 m=re.search(r'balance:\s*([0-9.]+)',usage)
 if not m or float(m[1])<158.4:
  log.open('a').write(json.dumps(dict(attempt=attempt,usage=usage,reason='budget gate'))+'\n');break
 p=subprocess.run(['colab','new','--gpu','L4','-s','tig039r-eval'],capture_output=True,text=True)
 row=dict(attempt=attempt,time=time.time(),usage=usage,returncode=p.returncode,output=p.stdout+p.stderr)
 log.open('a').write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
 if p.returncode==0:break
