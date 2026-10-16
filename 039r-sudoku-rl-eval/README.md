# Experiment 039r: E4B Sudoku GRPO replication

## Hypothesis and decision rule (written before execution)
H1: Gemma-4-E4B GRPO improves the expected fraction of five fixed Sudoku puzzles solved by a sampled strategy relative to its base model.
No original adapter is available locally; the original VM is gone. Retrain the original 16-bit rank-32 LoRA recipe, lr 5e-5, beta explicitly 0.001 (the original observed TRL default), 1024 completion tokens, with **4 completions per step and 24 steps** (original completed 20). This tests a strengthened replication, not the exact lost adapter.

Round 1: independently sample 64 strategies per arm, temperature 1, top-p .95, top-k 64, each scored on the original five seeds 910001–910005, difficulty 40. Primary unit is one strategy's mean solve fraction across five puzzles; never treat the 320 correlated puzzle outcomes as independent. Independent-arm cluster bootstrap, 100,000 resamples, deterministic analysis seed 39039. Report ordinary 95% CIs; decision uses 97.5% two-sided CIs at each of at most two looks (Bonferroni family coverage >=95%). PROVEN if lower bound of adapter minus base exceeds zero, DISPROVEN if upper bound is below zero, otherwise inconclusive. This does not test equivalence.

If inconclusive, redesign once by increasing to 128 strategies per arm (fresh additional 64 each; retain first 64), holding puzzles and generation settings fixed. Stop after round 2 even if STILL INCONCLUSIVE. Reserve compute for this contingency. A hard session deadline below 5 CU overrides all work; incomplete data never count as a settled result.

## Compute and provenance
Only session tig039r-eval will be used. Request L4; A100 only for demonstrated fit failure. Budget <=5 CU. Record usage before/after and actual GPU here after execution. No private data. Original folder remains untouched. 

## Resume audit (2026-10-15)
The earlier partial session ran 03:13:30–03:21:37 UTC on NVIDIA L4 (23,034 MiB), stopped explicitly; no model/evaluation artifacts were retained. Estimated prior spend 0.2085 CU (487.3 seconds × 1.54 CU/hour). Include this in the 5 CU cap. On resume, balance 176.94 CU with two unrelated active sessions; initial L4 allocation refused. Retry every five minutes up to six times, checking balance first. Do not stop unrelated sessions.
