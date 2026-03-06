# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## 2026-03-06 Legit Search Iteration

- Read `.ralph-tui/progress.md`, `.ralph-tui/policy_legit_controller.md`, and `eval/legit_completion_checklist.md` before evaluating any controller.
- `seed_shaping` remains invalid by policy and was ignored throughout this iteration.
- Policy-checked legitimate controllers:
  - `optimized`, `robust_v2`, `optimized_clipped`, `optimized_guarded`, `optimized_damped`, `optimized_warm`
  - `pid`, `preview`, `robust_simple`, `lqt_fixed`, `lqt_dob`, `model_inv`, `linear_mpc`, `bc_policy`, `onnx_1step`
  - `onnx_mppi_v2`, `onnx_mpc_v3`, `hybrid`, `hybrid_diff`, `diff_plan`, `model_corr`
  - All returned `POLICY OK`.

### Official Fast Gate Results

- `./eval/run.sh optimized 50 8 "42"` -> `anti_fuffa_score=56.943`
- `./eval/run.sh robust_v2 50 8 "42"` -> `anti_fuffa_score=56.943`
- `./eval/run.sh optimized_clipped 50 8 "42"` -> `anti_fuffa_score=52.736`
- `./eval/run.sh optimized_guarded 50 8 "42"` -> `anti_fuffa_score=67.218`
- `./eval/run.sh optimized_damped 50 8 "42"` -> `anti_fuffa_score=52.501`
- `./eval/run.sh optimized_warm 50 8 "42"` -> `anti_fuffa_score=54.059`
- `./eval/run.sh pid 50 8 "42"` -> `anti_fuffa_score=125.702`
- `./eval/run.sh preview 50 8 "42"` -> `anti_fuffa_score=851.442`
- `./eval/run.sh robust_simple 50 8 "42"` -> `anti_fuffa_score=57.825`
- `./eval/run.sh lqt_fixed 50 8 "42"` -> `anti_fuffa_score=13790.425`
- `./eval/run.sh lqt_dob 50 8 "42"` -> `anti_fuffa_score=107.838`
- `./eval/run.sh model_inv 50 8 "42"` -> `anti_fuffa_score=7548.293`
- `./eval/run.sh linear_mpc 50 8 "42"` -> `anti_fuffa_score=177.393`
- `./eval/run.sh bc_policy 50 8 "42"` -> `anti_fuffa_score=7161.558`
- `./eval/run.sh onnx_1step 50 8 "42"` -> `anti_fuffa_score=102249.476`

- No legitimate controller reached the medium gate threshold. The best official fast-gate result this iteration is still `optimized_damped` at `52.501`.

### Cheap Local Search And Diagnostics

- `./.venv/bin/python search_optimized_gate.py --num-segs 50 --eval-seed 42 --search-seed 42 --candidates 20 --workers 8`
  - Best sampled candidate: `rand_1`
  - Stats: `mean=53.508`, `p95=90.159`, `max=263.611`
  - Dropped: still worse than the best official fast-gate controller.
- Outlier inspection on `00254.csv`, `00432.csv`, `00480.csv`:
  - `optimized` costs: `538.346`, `1370.418`, `547.839`
  - `optimized_damped` costs: `449.215`, `227.611`, `526.032`
  - Pattern: `optimized_damped` helps on the positive high-target outlier (`00432`) but still saturates to `|lataccel|=5.0` on `00254` and `00480`.
- Unofficial model-based screen:
  - `./eval/run.sh onnx_mppi_v2 10 2 "42"`
  - Terminated after several minutes without returning a score. Current ONNX/MPC path is too slow for cheap screening in this workflow.

### Controller Experiments Dropped

- `optimized_tailguard`
  - Idea: add a high-lataccel same-direction recovery guard on top of `optimized_damped`.
  - Policy check: `POLICY OK`
  - Dropped after outlier regressions:
    - `00254.csv`: `449.215 -> 660.609`
    - `00432.csv`: `227.611 -> 2403.007`
    - `00480.csv`: `526.032 -> 779.805`
- `bc_policy_warm`
  - Idea: keep BC previous-action state neutral until step 100 because the simulator ignores controller actions during warmup.
  - Policy check: `POLICY OK`
  - Official fast gate: `./eval/run.sh bc_policy_warm 50 8 "42"` -> `anti_fuffa_score=6242.143`
  - Dropped: warmup mismatch was real but not enough to rescue the BC controller.
- `optimized_v5_short`
  - Idea: freeze params from a short representative-data CMA-ES run.
  - Policy check: `POLICY OK`
  - Official fast gate: `./eval/run.sh optimized_v5_short 50 8 "42"` -> `anti_fuffa_score=63.289`
  - Dropped: worse than the existing feedback variants.

### Representative-Data Optimization

- `./.venv/bin/python optimize_v5.py --workers 8 --segs 80 --valid-segs 120 --pop 8 --gen 4 --sigma 0.06`
  - Baseline on 80 uniformly spaced segments:
    - `mean=56.24`, `p95=92.71`, `max=508.96`, `obj=157.73`
  - Best training objective found:
    - `obj=131.01`
  - Holdout validation on 120 segments regressed versus baseline:
    - Candidate: `mean=52.61`, `p95=95.54`, `max=137.42`, `obj=82.45`
    - Baseline: `mean=46.93`, `p95=86.69`, `max=124.29`, `obj=74.33`
  - Result: candidate not promoted.

### Current Best Local Legitimate Result

- Controller: `optimized_damped`
- Policy check command: `python3 eval/check_controller_policy.py optimized_damped`
- Best official fast gate command: `./eval/run.sh optimized_damped 50 8 "42"`
- Best official fast gate `anti_fuffa_score`: `52.501`

### Iteration Status

- No legitimate controller proved `<43` on the fast gate.
- Because the fast gate never went below `43`, no controller was eligible for the medium or slow gate in this iteration.
- `eval/legit_completion_checklist.md` remains intentionally unfilled.
- Reopened the bead for the next iteration:
  - `bd update controls_challenge-2s2.1 --status open --json`
  - Result: status is now `open`.
