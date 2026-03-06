# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## 2026-03-06 - controls_challenge-2s1.1

### Findings

- `eval/best_score.json` is stale for gating purposes. It records `optimized` on only `3` segments per seed and `2` seeds (`num_segs_per_seed=3`, `num_seeds=2`), so its `anti_fuffa_score=37.117` is not comparable to the required `50 -> 100 -> 5000` pipeline.
- `controllers/seed_shaping.py` is invalid. `python3 eval/check_controller_policy.py seed_shaping` fails for frame introspection, simulator-history reads, and RNG reseeding.
- The strongest valid fast-gate controller found in this iteration is still `optimized_damped`, but it is not close enough to promote:
  - `./eval/run.sh optimized_damped 50 8 "42"` -> `score.anti_fuffa=52.501`
  - `./eval/run.sh optimized_clipped 50 8 "42"` -> `score.anti_fuffa=52.736`
  - `./eval/run.sh optimized 50 8 "42"` -> `score.anti_fuffa=56.943`
- Hand-tuned or rebuilt feedback variants tested here stayed above the current best:
  - `./eval/run.sh optimized_warm 50 8 "42"` -> `54.059`
  - `./eval/run.sh robust_simple 50 8 "42"` -> `57.825`
  - `./eval/run.sh robust_v2 50 8 "42"` -> `56.943`
  - `./eval/run.sh preview 50 8 "42"` -> `851.442`
  - `./eval/run.sh pid 50 8 "42"` -> `125.702`
  - `./eval/run.sh model_inv 50 8 "42"` -> `7548.293`
  - `./eval/run.sh lqt_fixed 50 8 "42"` -> `13790.425`
- `bc_policy` was initially blocked by a broken training path. The dataset only has finite `steerCommand` labels before the control handoff, so the original extractor was training on all-`NaN` labels. After fixing the extractor and training a checkpoint, the rollout was still unusable:
  - `./.venv/bin/python train_bc_policy.py --num-segs 5000 --epochs 5 --batch-size 8192 --lr 1e-3`
  - `./eval/run.sh bc_policy 50 8 "42"` -> `7161.558`
- Linear feedforward regression and a hand-written linear MPC prototype were screened off-line in ad hoc scripts and were much worse than the `optimized_*` family; they were not promoted to controller files.
- `tdmpc2_ctrl` is not a hidden win on the stored checkpoints. `PYTHONPATH=. ./.venv/bin/python tdmpc2/diagnose.py --num_segs 2` showed:
  - `ep50` mean total cost `1056.98`
  - `ep100` mean total cost `1535.74`
  - `eval_best` mean total cost `1056.98`

### Commands Run

- Policy checks:
  - `python3 eval/check_controller_policy.py optimized`
  - `python3 eval/check_controller_policy.py optimized_guarded`
  - `python3 eval/check_controller_policy.py optimized_clipped`
  - `python3 eval/check_controller_policy.py robust_simple`
  - `python3 eval/check_controller_policy.py lqt_fixed`
  - `python3 eval/check_controller_policy.py seed_shaping`
  - `python3 eval/check_controller_policy.py onnx_1step`
  - `python3 eval/check_controller_policy.py onnx_mpc_v3`
  - `python3 eval/check_controller_policy.py onnx_mppi_v2`
  - `python3 eval/check_controller_policy.py hybrid`
  - `python3 eval/check_controller_policy.py hybrid_diff`
  - `python3 eval/check_controller_policy.py tdmpc2_ctrl`
  - `python3 eval/check_controller_policy.py model_inv`
  - `python3 eval/check_controller_policy.py bc_policy`
  - `python3 eval/check_controller_policy.py optimized_warm`
- Fast-gate runs:
  - `./eval/run.sh optimized 50 8 "42"`
  - `./eval/run.sh optimized_clipped 50 8 "42"`
  - `./eval/run.sh optimized_damped 50 8 "42"`
  - `./eval/run.sh optimized_guarded 50 8 "42"`
  - `./eval/run.sh optimized_tuned 50 8 "42"`
  - `./eval/run.sh robust_simple 50 8 "42"`
  - `./eval/run.sh robust_v2 50 8 "42"`
  - `./eval/run.sh lqt_fixed 50 8 "42"`
  - `./eval/run.sh preview 50 8 "42"`
  - `./eval/run.sh pid 50 8 "42"`
  - `./eval/run.sh model_inv 50 8 "42"`
  - `./eval/run.sh bc_policy 50 8 "42"`
  - `./eval/run.sh optimized_warm 50 8 "42"`
- Search / diagnostics:
  - `./.venv/bin/python search_optimized_gate.py --num-segs 50 --eval-seed 42 --search-seed 42 --candidates 20 --workers 8`
  - `./.venv/bin/python fit_linear_model.py`
  - `./.venv/bin/python train_bc_policy.py --num-segs 1000 --epochs 4 --batch-size 8192 --lr 1e-3`
  - `./.venv/bin/python train_bc_policy.py --num-segs 5000 --epochs 5 --batch-size 8192 --lr 1e-3`
  - `PYTHONPATH=. ./.venv/bin/python tdmpc2/diagnose.py --num_segs 2`

### Code Changes

- Added `controllers/optimized_warm.py`:
  - warmup-aware variant of the best current feedback family
  - keeps dynamic state neutral until the first real control step so ignored pre-control actions do not poison rate-limited state
- Fixed `train_bc_policy.py`:
  - extracts only rows with finite logged `steerCommand` labels instead of the post-handoff `NaN` region

### Current Best Local Score

- Best valid fast-gate result in this iteration: `optimized_damped`
- Command: `./eval/run.sh optimized_damped 50 8 "42"`
- Score: `anti_fuffa_score=52.501`

### Status

- No valid controller in this iteration achieved the required medium-gate entry condition (`anti_fuffa_score < 43` on the required pipeline), so the task remains open.
