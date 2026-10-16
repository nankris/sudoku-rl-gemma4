# launched on the Colab VM via `colab exec -s tig039 -f launch.py`; runs the notebook headless in the background
import subprocess, os
os.makedirs('/content/results', exist_ok=True)
p = subprocess.Popen("cd /content && nohup jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 "
                     "sudoku_grpo_039.ipynb --output sudoku_grpo_039_output.ipynb > /content/results/nbconvert.log 2>&1 &",
                     shell=True)
print("launched")
