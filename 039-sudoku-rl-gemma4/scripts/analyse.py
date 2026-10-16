"""#039 analysis: base vs GRPO-trained on the fixed Sudoku eval, for Gemma 4 E2B and E4B (both L4, 20 GRPO steps),
plus the partial T4 E2B reward curve. Writes chart.png and results/summary_table.json."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RUNS = {"Gemma 4 E2B": "e2b_l4/results", "Gemma 4 E4B": "e4b_l4/results"}
def load(d):
    d = os.path.join(R, d)
    if not os.path.exists(f"{d}/eval_trained.json"): return None
    logs = [json.loads(l) for l in open(f"{d}/train_log.jsonl")]
    steps = [l for l in logs if "reward" in l]
    hist = json.load(open(f"{d}/train_history.json"))
    return {"steps": steps, "base": json.load(open(f"{d}/eval_base.json"))["summary"],
            "trained": json.load(open(f"{d}/eval_trained.json"))["summary"], "train_seconds": hist["train_seconds"],
            "global_step": hist["global_step"]}
data = {k: load(v) for k, v in RUNS.items()}; data = {k: v for k, v in data.items() if v}
t4 = json.load(open(f"{R}/t4_e2b_run_partial.json"))
def ma(v, k=5): return [sum(v[max(0, i-k+1):i+1]) / len(v[max(0, i-k+1):i+1]) for i in range(len(v))]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
cols = {"Gemma 4 E2B": "#2a6", "Gemma 4 E4B": "#36c"}
for k, v in data.items():
    x = [s["step"] for s in v["steps"]]; y = [s["reward"] for s in v["steps"]]
    a.plot(x, y, color=cols[k], alpha=.25, lw=1); a.plot(x, ma(y), color=cols[k], lw=2, label=f"{k}, L4 (5-step avg)")
ty = t4["train_reward_steps_1_21"]
a.plot(range(1, len(ty)+1), ma(ty), color="#2a6", ls=":", lw=1.5, label="Gemma 4 E2B, T4 run (lost at step ~27)")
a.axhline(32, color="#aaa", ls="--", lw=.8); a.text(0.6, 32.8, "max reward 32 (solve + valid + no-cheat)", fontsize=7, color="#777")
a.set_xlabel("GRPO step (2 completions per step)"); a.set_ylabel("total reward"); a.set_ylim(-15, 38)
a.set_title("#039 Sudoku-strategy GRPO: reward per step", fontsize=10); a.legend(fontsize=7, loc="lower right")
labels, bars = [], []
for k, v in data.items():
    labels += [f"{k.split()[-1]}\nbase", f"{k.split()[-1]}\nafter GRPO"]
    bars += [(100*v["base"]["solve_rate"], "#bbb"), (100*v["trained"]["solve_rate"], cols[k])]
b.bar(range(len(bars)), [x[0] for x in bars], color=[x[1] for x in bars])
for i, (val, _) in enumerate(bars): b.text(i, val + 1.5, f"{val:.1f}%", ha="center", fontsize=9)
b.set_xticks(range(len(labels))); b.set_xticklabels(labels, fontsize=8); b.set_ylim(0, 100)
b.set_ylabel("% of (strategy, puzzle) pairs solved"); b.set_title("Fixed eval: 16 sampled strategies x 5 puzzles", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(R, "..", "chart.png"), dpi=130)
out = {}
for k, v in data.items():
    y = [s["reward"] for s in v["steps"]]
    out[k] = {"gpu": "L4", "grpo_steps": v["global_step"], "train_minutes": round(v["train_seconds"]/60, 1),
              "sec_per_step": round(v["train_seconds"]/max(1, v["global_step"]), 1),
              "reward_first5_mean": round(sum(y[:5])/5, 2), "reward_last5_mean": round(sum(y[-5:])/5, 2),
              "base": v["base"], "trained": v["trained"]}
json.dump(out, open(f"{R}/summary_table.json", "w"), indent=1); print(json.dumps(out, indent=1))
