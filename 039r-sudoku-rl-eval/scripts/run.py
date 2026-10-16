import sys, subprocess, os
# A standalone runner must not call Colab notebook secret widgets.
sys.modules['google.colab'] = None
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
def install(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', *args], check=True)
install('unsloth', 'unsloth_zoo', 'transformers>=5.5.0,<5.16', 'tokenizers>=0.22.0,<=0.23.0', 'trl>=0.28.0', 'huggingface_hub>=1.5.0,<2.0', 'timm', 'bitsandbytes', 'nbconvert')


# Cell 0
# #039 config
import os, time, json
os.makedirs('/content/results', exist_ok=True)
MAX_STEPS = 24
BETA = 0.001
TRAIN_MINUTES = 125.0
T0 = time.time()
def plog(msg):
    with open('/content/results/progress.log','a') as f:
        f.write(f"[{time.time()-T0:7.0f}s] {msg}\n")
plog('start')


# Cell 3
plog('installed')

# Cell 4
from unsloth import FastVisionModel
import torch
max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-4-E4B-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = False, # Enable vllm fast inference
)
plog('model loaded')

# Cell 5
model = FastVisionModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

# Cell 6
#@title Sudoku Game Implementation
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import copy

def _is_valid_placement(board: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if placing num at (row, col) is valid."""
    # Check row
    if num in board[row]:
        return False

    # Check column
    if num in [board[r][col] for r in range(9)]:
        return False

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False

    return True

def _solve_sudoku(board: List[List[int]]) -> bool:
    """Solve sudoku using backtracking (for puzzle generation)."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if _is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if _solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def _generate_complete_board(rng: random.Random) -> List[List[int]]:
    """Generate a complete valid Sudoku board."""
    board = [[0 for _ in range(9)] for _ in range(9)]

    # Fill diagonal 3x3 boxes first (they don't affect each other)
    for box in range(3):
        nums = list(range(1, 10))
        rng.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]

    # Solve the rest
    _solve_sudoku(board)
    return board

@dataclass
class SudokuGame:
    difficulty: int = 40  # Number of cells to remove (20 = easy, 40 = medium, 50 = hard)
    seed: Optional[int] = None
    _rng: random.Random = field(init = False, repr = False)
    _board: List[List[int]] = field(init = False, repr = False)
    _solution: List[List[int]] = field(init = False, repr = False)
    _initial_board: List[List[int]] = field(init = False, repr = False)
    _moves: int = field(default = 0, init = False, repr = False)
    _state: str = field(default = "ongoing", init = False, repr = False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

        # Generate complete board
        complete_board = _generate_complete_board(self._rng)
        self._solution = copy.deepcopy(complete_board)

        # Remove cells to create puzzle
        self._board = copy.deepcopy(complete_board)
        cells = [(r, c) for r in range(9) for c in range(9)]
        self._rng.shuffle(cells)

        for r, c in cells[:self.difficulty]:
            self._board[r][c] = 0

        self._initial_board = copy.deepcopy(self._board)
        self._update_state()

    def board(self) -> List[List[int]]:
        """Return current board state."""
        return [row[:] for row in self._board]

    def initial_board(self) -> List[List[int]]:
        """Return initial puzzle state."""
        return [row[:] for row in self._initial_board]

    def state(self) -> str:
        """Return game state: 'ongoing', 'success', or 'failed'."""
        return self._state

    def moves(self) -> int:
        """Return number of moves made."""
        return self._moves

    def place_number(self, row: int, col: int, num: int) -> bool:
        """Place a number on the board. Returns True if valid move."""
        # Validate input
        if not (0 <= row < 9 and 0 <= col < 9):
            self._state = "failed"
            return False

        if not (1 <= num <= 9):
            self._state = "failed"
            return False

        # Can't modify initial cells
        if self._initial_board[row][col] != 0:
            self._state = "failed"
            return False
        if self._board[row][col] != 0:
            self._state = "failed"
            return False
        # Check if placement is valid
        if not _is_valid_placement(self._board, row, col, num):
            self._state = "failed"
            return False

        # Place number
        self._board[row][col] = num
        self._moves += 1
        self._update_state()
        return True

    def _update_state(self) -> None:
        """Update game state based on current board."""
        # Check if puzzle is complete
        if all(self._board[r][c] != 0 for r in range(9) for c in range(9)):
            # Verify solution is correct
            if self._board == self._solution:
                self._state = "success"
            else:
                self._state = "failed"
        else:
            self._state = "ongoing"

    def pretty(self, colors: bool = True) -> str:
        """Pretty print the Sudoku board."""
        RESET = "\x1b[0m"
        INITIAL = "\x1b[38;5;45m"   # Cyan for initial numbers
        PLACED = "\x1b[38;5;226m"    # Yellow for placed numbers
        EMPTY = "\x1b[38;5;239m"     # Gray for empty cells

        lines = []
        lines.append("┌───────┬───────┬───────┐")

        for row in range(9):
            row_str = "│ "
            for col in range(9):
                num = self._board[row][col]

                if colors:
                    if num == 0:
                        row_str += f"{EMPTY}.{RESET}"
                    elif self._initial_board[row][col] != 0:
                        row_str += f"{INITIAL}{num}{RESET}"
                    else:
                        row_str += f"{PLACED}{num}{RESET}"
                else:
                    row_str += str(num) if num != 0 else "."

                if col % 3 == 2:
                    row_str += " │ "
                else:
                    row_str += " "

            lines.append(row_str.rstrip())

            if row == 8:
                lines.append("└───────┴───────┴───────┘")
            elif row % 3 == 2:
                lines.append("├───────┼───────┼───────┤")

        return "\n".join(lines)

# Cell 7
from typing import Callable
from unsloth import execute_with_time_limit

def _execute_strategy(strategy: Callable, game: SudokuGame):
    """Execute a strategy function on a Sudoku game."""
    assert callable(strategy)

    max_moves = 100
    valid_moves = 0  # Track successful moves

    while game.state() == "ongoing" and valid_moves < max_moves:
        try:
            board = game.board()
            initial = game.initial_board()
            result = strategy(board, initial)

            # Validate result format
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                # Invalid format = immediate fail, but return valid moves made
                return valid_moves, "failed"

            row, col, num = result

            # Validate types
            if not all(isinstance(x, int) for x in [row, col, num]):
                return valid_moves, "failed"

            # Try to place number
            success = game.place_number(row, col, num)

            if success:
                valid_moves += 1  # Count this valid move
            else:
                # Invalid move = game fails, but return valid_moves made so far
                return valid_moves, "failed"

        except Exception:
            return valid_moves, "failed"

    if valid_moves >= max_moves and game.state() == "ongoing":
        return valid_moves, "failed"

    return valid_moves, game.state()

# Cell 8
@execute_with_time_limit(10)
def execute_strategy(strategy: Callable, game: SudokuGame):
    """Execute strategy with 10 second time limit."""
    return _execute_strategy(strategy, game)

# Cell 9
from unsloth import check_python_modules, create_locked_down_function

# Cell 10
prompt = """
Create a Sudoku solving strategy using only native Python built-in functions without any import statements.
You are given two lists of lists (9x9 grids):
- board: current state (0 means empty)
- initial: starting puzzle (0 means was empty, numbers are fixed)

Return a tuple (row, col, number) for the next move.
- row: 0-8 (row index)
- col: 0-8 (column index)
- number: 1-9 (digit to place)

Only place numbers in cells that are BOTH empty in initial AND empty in board (initial[row][col] == 0 AND board[row][col] == 0)
Use Sudoku rules: no duplicates in rows, columns, or 3x3 boxes.
Output your function in backticks:
```python
def strategy(board, initial):
    # Your logic here
    return (row, col, number)
```
All helper functions must be inside def strategy. Output only the function.
""".strip()

print(prompt)

# Cell 11
def extract_function(text):
    """Extract Python function from markdown code blocks."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board, initial):"):
            return fx
    return None

# Cell 12
def function_works(completions, **kwargs):
    """Reward for generating valid executable Python code."""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)

        if function is not None:
            ok, info = check_python_modules(function)

        if function is None or "error" in info:
            score = -2.0  # Invalid function
        else:
            try:
                new_strategy = create_locked_down_function(function)
                score = 1.0  # Valid function
            except:
                score = -1.0  # Function has errors

        scores.append(score)
    return scores

# Cell 13
def no_cheating(completions, **kwargs):
    """Penalize use of external imports."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)

        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)  # Heavy penalty for cheating
        else:
            scores.append(-1.0)  # Failed to create function

    return scores

# Cell 14
import numpy as np

global PRINTER
PRINTER = 0

def strategy_succeeds(completions, **kwargs):
    """Reward valid moves even if strategy eventually fails."""
    global PRINTER
    scores = []

    seed = np.random.randint(10000)
    difficulty = 40
    for completion in completions:
        printed = False
        response = completion[0]["content"]
        function = extract_function(response)

        if PRINTER % 5 == 0:
            printed = True
            print("\n" + "=" * 60)
            print(function)
            print("=" * 60)
        PRINTER += 1

        if function is not None:
            ok, info = check_python_modules(function)

        if function is None or "error" in info:
            scores.append(0)
            continue

        try:
            new_strategy = create_locked_down_function(function)
        except:
            scores.append(0)
            continue

        try:
            game = SudokuGame(difficulty = difficulty, seed = seed)
            valid_moves, game_state = execute_strategy(new_strategy, game)
            if valid_moves == difficulty:
                game_state = "success"

            print(f"\n Valid moves: {valid_moves}, Final state: {game_state}")

            if not printed:
                print("Strategy:")
                print(function[:200] + "..." if len(function) > 200 else function)

            print("\nFinal board:")
            print(game.pretty())

            if game_state == "success":
                scores.append(30.0)  # Solved the puzzle!
            elif valid_moves > 0:
                # Reward based on valid moves made before failure
                # Each valid move is worth 0.2 points
                reward = valid_moves * 0.2
                scores.append(reward)
            else:
                scores.append(-2.0)  # Failed immediately with no valid moves

        except TimeoutError:
            print("Timeout")
            scores.append(-1.0)
        except Exception as e:
            print(f"Exception: {str(e)[:100]}")
            scores.append(-3.0)

    return scores

# Cell 15
# @title #039 fixed Sudoku eval (added; same for base and trained)
import torch, contextlib, io
EVAL_SEEDS = [910001, 910002, 910003, 910004, 910005]   # fixed puzzles, difficulty 40 (same as training reward)
EVAL_SAMPLES = 64      # strategies sampled per model
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


# Cell 16
base_eval = run_eval('base')
print(base_eval)

# Cell 17
from datasets import Dataset

dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": prompt.strip()}],
        "answer": 0,
    }
] * 1000)

maximum_length = len(tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt.strip()}],
    add_generation_prompt = True
))

print(f"Maximum prompt length: {maximum_length}")
print("\nDataset sample:")
print(dataset[0])

# Cell 18
# Leave room for the prompt (plus 1 token safety margin)
max_completion_length = min(1024, max_seq_length - (maximum_length + 1))  # #039: capped for T4 budget

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = MAX_STEPS,
    save_steps = 100,
    report_to = "none", # Can use Weights & Biases, TrackIO
    output_dir = "outputs",
    epsilon = 0.2,
    epsilon_high = 0.28, # one sided
    delta = 1.5, # two sided
    loss_type = 'bnpo',
    mask_truncated_completions = True,
    **({} if BETA is None else {'beta': BETA}),
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)
print('beta =', training_args.beta, 'max_completion_length =', training_args.max_completion_length)
plog(f'beta={training_args.beta} mcl={training_args.max_completion_length}')

# Cell 19
# @title progress + wall-clock budget callback (added)
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


# Cell 20
# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        function_works,
        no_cheating,
        strategy_succeeds,
    ],
    args = training_args,
    callbacks = [BudgetCB()],
    train_dataset = dataset,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

# Cell 21
plog('train start')
_tt=time.time()
train_out = trainer.train()
train_seconds=time.time()-_tt
plog(f'train done {train_seconds:.0f}s')
json.dump({'log_history': trainer.state.log_history, 'train_seconds': train_seconds, 'global_step': trainer.state.global_step, 'metrics': train_out.metrics}, open('/content/results/train_history.json','w'), indent=1)

# Cell 22
model.save_pretrained("gemma_4_lora")  # Local saving
tokenizer.save_pretrained("gemma_4_lora")
import shutil
shutil.make_archive("/content/results/adapter", "zip", "/content/gemma_4_lora")

# Cell 23
trained_eval = run_eval('trained', seed=2345)
print(trained_eval)
json.dump({'base': base_eval, 'trained': trained_eval, 'total_seconds': time.time()-T0}, open('/content/results/summary.json','w'), indent=1)
plog('ALL DONE')
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
