import subprocess,time,json
from pathlib import Path
Path('/content/results').mkdir(exist_ok=True)
Path('/content/results/start_epoch.txt').write_text(str(time.time()))
p=subprocess.Popen(['jupyter','nbconvert','--to','notebook','--execute','--ExecutePreprocessor.timeout=-1','/content/run.ipynb','--output','/content/output.ipynb'],stdout=open('/content/results/run.log','w'),stderr=subprocess.STDOUT)
# Hard stop at 185 minutes (<4.75 CU at 1.54/h). Caller also stops VM.
subprocess.Popen(['bash','-c',f'sleep 11100; kill {p.pid} 2>/dev/null'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print('launched',p.pid)
