# Writeup: #039 Can 20 steps of RL teach a small Gemma to write Sudoku solvers?

**The task.** Instead of solving Sudoku directly, the model writes a short Python strategy function, which a game harness then runs. GRPO rewards it when the function runs, when it passes the notebook's no-cheating check, and when the strategy actually solves the puzzle. This is Unsloth's Gemma 4 Sudoku RL notebook, cut down to 20 steps with LoRA rank 32 so it fits a ~2 CU Colab budget.

**E4B learns, E2B mostly doesn't.** On a fixed eval (16 sampled strategies × 5 puzzles, same before and after), Gemma 4 E4B went from **50% to 68.75%** solved, and from 27.8 to 38.0 valid moves out of 40. That is a clear improvement from about 50 minutes of training on one L4. Gemma 4 E2B started at **6.25%** and stayed at 6.25%. It learned to make more valid moves (24.6 → 29.4), but its functions got longer, and one in 16 no longer ran. The reward curve shows the same thing. E2B's 5-step average drifted between 7 and 20 and ended at 7.2. E4B dipped early, recovered to the mid-20s, and ended around 16, starting from a higher base. With only 2 completions per step the per-step reward is very noisy (it swings between 0 and the 32 maximum), so only the eval numbers count.

**Why the difference.** With 2 completions per prompt, GRPO learns only when one completion beats the other. E4B already solves about half the puzzles, so it often gets a winner and a loser to compare. E2B rarely solves anything, so most steps give it little to learn from (our reading, not tested). More steps or more completions per step would be the next thing to try for E2B.

**What went wrong.** The first E2B run was on a T4 with a 100-step plan. Colab ended the session server-side after 100 minutes, around step 27, before any results were saved off the VM, so 1.78 CU went to a run with no trained eval. Its base eval and early reward curve survive from progress polls. Both final runs were moved to an L4, where a step took 110 s (E2B) and 154 s (E4B) against ~145 s for E2B on the T4. Each run had a wall-clock cap so it would finish inside its session.

**Compute.** T4 tig039 ~1.78 CU (lost), L4 tig039b (E4B) 68.7 min ~1.76 CU, L4 tig039c (E2B) 49.6 min ~1.27 CU. Total ~4.81 CU.

**Caveats.** One run per model, a 16 × 5 eval (each point is 1/16 of the solve rate), 19–20 steps, and E2B's T4 and L4 base evals differ (27.8 vs 24.6 valid moves) only from sampling.
