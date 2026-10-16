"""Build the modified Unsloth Gemma4 E2B Sudoku GRPO notebook for #039.
Changes vs orig.ipynb (Unsloth notebooks repo, nb/Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb):
 - drop demo/markdown cells and the export/GGUF cells
 - max_steps = MAX_STEPS (env-free constant below), max_completion_length capped at 1024
 - BETA constant (None = TRL default; 0.0 fallback)
 - add a fixed Sudoku eval (same before and after training), a progress log, a wall-clock stop callback
 - save log_history + eval results as JSON in /content/results/
"""
import json, sys, copy
MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BETA = sys.argv[2] if len(sys.argv) > 2 else "None"
TRAIN_MINUTES = float(sys.argv[3]) if len(sys.argv) > 3 else 90
MODEL = sys.argv[4] if len(sys.argv) > 4 else "unsloth/gemma-4-E2B-it"
LOAD4 = (sys.argv[5] == "4bit") if len(sys.argv) > 5 else False
OUTNB = sys.argv[6] if len(sys.argv) > 6 else "sudoku_grpo_039.ipynb"
orig = json.load(open("orig.ipynb"))
cells = orig["cells"]
def src(i): return "".join(cells[i]["source"])
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

new = []
new.append(code(f"""# #039 config
import os, time, json
os.makedirs('/content/results', exist_ok=True)
MAX_STEPS = {MAX_STEPS}
BETA = {BETA}
TRAIN_MINUTES = {TRAIN_MINUTES}
T0 = time.time()
def plog(msg):
    with open('/content/results/progress.log','a') as f:
        f.write(f"[{{time.time()-T0:7.0f}}s] {{msg}}\\n")
plog('start')
"""))
new.append(code(src(3)))           # installs
new.append(code(src(4)))           # timm
new.append(code("plog('installed')"))
m6 = src(6).replace('model_name = "unsloth/gemma-4-E2B-it"', f'model_name = "{MODEL}"').replace("load_in_4bit = False", f"load_in_4bit = {LOAD4}")
assert MODEL in m6
new.append(code(m6 + "\nplog('model loaded')"))
new.append(code(src(8)))           # peft
new.append(code(src(10)))          # sudoku game
new.append(code(src(18)))          # _execute_strategy
new.append(code(src(20)))          # time limit
new.append(code("from unsloth import check_python_modules, create_locked_down_function"))
new.append(code(src(29)))          # prompt
new.append(code(src(33)))          # extract_function
new.append(code(src(35)))
new.append(code(src(37)))
new.append(code(src(39)))
# ---- eval cell (added) ----
new.append(code(r'''# @title #039 fixed Sudoku eval (added; same for base and trained)
import torch, contextlib, io
EVAL_SEEDS = [910001, 910002, 910003, 910004, 910005]   # fixed puzzles, difficulty 40 (same as training reward)
EVAL_SAMPLES = 16      # strategies sampled per model
EVAL_BATCH = 8
EVAL_MAX_NEW = 1024
def score_function(function):
    """Mirror of strategy_succeeds scoring, per puzzle. Returns dict."""
    if function is None: return {"valid_fn": False, "moves": [0]*len(EVAL_SEEDS), "solved": [False]*len(EVAL_SEEDS), "reward": [0.0]*len(EVAL_SEEDS)}
    ok, info = check_python_modules(function)
    if "error" in info or not ok:
        return {"valid_fn": False, "moves": [0]*len(EVAL_SEEDS), "solved": [False]*len(EVAL_SEEDS), "reward": [0.0]*len(EVAL_SEEDS)}
    try:
        strat = create_locked_down_function(function)
    except Exception:
        return {"valid_fn": False, "moves": [0]*len(EVAL_SEEDS), "solved": [False]*len(EVAL_SEEDS), "reward": [0.0]*len(EVAL_SEEDS)}
    moves, solved, rew = [], [], []
    for s in EVAL_SEEDS:
        g = SudokuGame(difficulty=40, seed=s)
        try:
            vm, st = execute_strategy(strat, g)
            if vm == 40: st = "success"
            r = 30.0 if st == "success" else (vm*0.2 if vm > 0 else -2.0)
        except TimeoutError:
            vm, st, r = 0, "timeout", -1.0
        except Exception:
            vm, st, r = 0, "error", -3.0
        moves.append(vm); solved.append(st == "success"); rew.append(r)
    return {"valid_fn": True, "moves": moves, "solved": solved, "reward": rew}

def run_eval(tag, seed=1234):
    t = time.time()
    FastVisionModel.for_inference(model)
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt.strip()}], tokenize=False, add_generation_prompt=True)
    torch.manual_seed(seed)
    outs, ntoks = [], []
    for b in range(EVAL_SAMPLES // EVAL_BATCH):
        inputs = tokenizer(text=[text]*EVAL_BATCH, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=EVAL_MAX_NEW, do_sample=True, use_cache=True,
                                 temperature=1.0, top_p=0.95, top_k=64)
        L = inputs["input_ids"].shape[1]
        for row in gen[:, L:]:
            ids = row.tolist()
            pad = tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, "tokenizer") else tokenizer.pad_token_id
            ntoks.append(sum(1 for x in ids if x != pad))
            outs.append(tokenizer.decode(row, skip_special_tokens=True))
        plog(f"eval {tag}: batch {b} done")
    gen_s = time.time() - t
    rows = []
    for o, n in zip(outs, ntoks):
        fn = extract_function(o)
        with contextlib.redirect_stdout(io.StringIO()):
            sc = score_function(fn)
        rows.append({"completion": o, "n_tokens": n, "function_found": fn is not None, **sc})
    N = len(rows); P = len(EVAL_SEEDS)
    summ = {
        "tag": tag, "n_samples": N, "n_puzzles": P,
        "valid_function_rate": sum(r["valid_fn"] for r in rows)/N,
        "solve_rate": sum(sum(r["solved"]) for r in rows)/(N*P),
        "strategies_solving_all": sum(all(r["solved"]) for r in rows)/N,
        "mean_valid_moves_of_40": sum(sum(r["moves"]) for r in rows)/(N*P),
        "mean_reward_strategy_succeeds": sum(sum(r["reward"]) for r in rows)/(N*P),
        "mean_completion_tokens": sum(ntoks)/N,
        "eval_seconds": time.time()-t, "gen_seconds": gen_s,
    }
    json.dump({"summary": summ, "rows": rows}, open(f"/content/results/eval_{tag}.json", "w"), indent=1)
    plog(f"eval {tag}: {json.dumps(summ)}")
    FastVisionModel.for_training(model)
    return summ
'''))
new.append(code("base_eval = run_eval('base')\nprint(base_eval)"))
new.append(code(src(41)))
cfg = src(43).replace("max_completion_length = max_seq_length - (maximum_length + 1)",
                      "max_completion_length = min(1024, max_seq_length - (maximum_length + 1))  # #039: capped for T4 budget")
cfg = cfg.replace("max_steps = 60,", "max_steps = MAX_STEPS,")
cfg = cfg.replace("mask_truncated_completions = True", "mask_truncated_completions = True,\n    **({} if BETA is None else {'beta': BETA}),")
assert "MAX_STEPS" in cfg and "min(1024" in cfg
new.append(code(cfg + "\nprint('beta =', training_args.beta, 'max_completion_length =', training_args.max_completion_length)\nplog(f'beta={training_args.beta} mcl={training_args.max_completion_length}')"))
new.append(code(r'''# @title progress + wall-clock budget callback (added)
from transformers import TrainerCallback
class BudgetCB(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        self.t = time.time()
    def on_log(self, args, state, control, logs=None, **kw):
        logs = dict(logs or {}); logs["step"] = state.global_step; logs["elapsed_s"] = time.time()-self.t
        with open('/content/results/train_log.jsonl','a') as f: f.write(json.dumps(logs)+"\n")
        plog(f"step {state.global_step} reward={logs.get('reward')} len={logs.get('completions/mean_length')}")
    def on_step_end(self, args, state, control, **kw):
        if time.time()-self.t > TRAIN_MINUTES*60:
            plog(f"budget stop at step {state.global_step}"); control.should_training_stop = True
'''))
new.append(code(src(45).replace("args = training_args,", "args = training_args,\n    callbacks = [BudgetCB()],")))
new.append(code("plog('train start')\n_tt=time.time()\ntrain_out = trainer.train()\ntrain_seconds=time.time()-_tt\nplog(f'train done {train_seconds:.0f}s')\njson.dump({'log_history': trainer.state.log_history, 'train_seconds': train_seconds, 'global_step': trainer.state.global_step, 'metrics': train_out.metrics}, open('/content/results/train_history.json','w'), indent=1)"))
new.append(code(src(49)))
new.append(code("trained_eval = run_eval('trained')\nprint(trained_eval)\njson.dump({'base': base_eval, 'trained': trained_eval, 'total_seconds': time.time()-T0}, open('/content/results/summary.json','w'), indent=1)\nplog('ALL DONE')"))
nb = {"cells": new, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(OUTNB, "w"), indent=1)
print("cells:", len(new))
