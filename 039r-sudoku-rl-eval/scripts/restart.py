import subprocess,os,signal
# Stop only processes belonging to our own notebook run in this dedicated VM.
out=subprocess.getoutput('ps -eo pid,args')
for line in out.splitlines():
 if 'nbconvert' in line and '/content/run.ipynb' in line and not 'ps -eo' in line:
  try: os.kill(int(line.strip().split()[0]),signal.SIGTERM)
  except ProcessLookupError: pass
print('stopped initial setup run before evaluation/training')
