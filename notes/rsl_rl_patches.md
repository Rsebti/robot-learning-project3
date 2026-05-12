# rsl_rl upstream patches

This project applies **one** local modification to the rsl_rl package
installed in the venv at
`C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Lib/site-packages/rsl_rl/`.

The patch is NOT tracked by git (it lives in site-packages, not in the
repo). It is re-applied by `sim/eval2/scripts/patch_rsl_rl.py`.

## Patch 1 — LR floor 1e-5 → 1e-4

**File** : `rsl_rl/algorithms/ppo.py`, line ~282

**Before** :
```python
if kl_mean > self.desired_kl * 2.0:
    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
```

**After** :
```python
if kl_mean > self.desired_kl * 2.0:
    # PATCHED 2026-05-12 by robot-learning-project3 V2.18b ...
    self.learning_rate = max(1e-4, self.learning_rate / 1.5)
```

### Why

V2.18 cold-start (run `2026-05-11_23-32-42`) collapsed by iter 200 to
`learning_rate = 1e-5` (the rsl_rl hard-coded floor) under adaptive-KL
throttling. The optimizer stayed frozen there for 215 iters; mean_reward
plateaued at ~14, then a single bad minibatch at iter 415 triggered
value-function blowup (0.009 → 68 → 1.3e9 → inf in 5 iters), killing
the run.

Raising the floor to 1e-4 means :
- adaptive throttling can still respond to KL overshoots (3 halvings
  from init 3e-4 → 6e-5 ... wait, it floors at 1e-4 first)
- after init 3e-4 there are exactly 3 throttle steps before the floor
  (3e-4 → 2e-4 → 1.3e-4 → 1e-4), then it stays at 1e-4 indefinitely
- the optimizer remains responsive when the policy needs to escape a
  local optimum (e.g., when sparse grasp signal finally fires)

Source for the recommendation : `notes/v218b_ppo_search_claude.md`,
priority item #2.

### When to re-apply

Run `python sim/eval2/scripts/patch_rsl_rl.py` after any of :
- venv rebuild (`uv sync`, `pip install -r requirements.txt`, etc.)
- `pip install --upgrade rsl-rl-lib`
- `pip install --force-reinstall isaac_so_arm101` (transitive)
- cloning the repo onto a new machine

The script is **idempotent** — it exits 0 with `[OK] already patched`
if the patch is in place, otherwise applies it.

### Risk if forgotten

Without the patch, the LR can collapse to 1e-5 under sustained KL
overshoots (typically iter 100-300 of any cold-start). The training
will appear to make slow progress for a while (frozen optimizer ≠ no
training, just very slow), then a sufficiently spiky advantage update
detonates VF and the run dies. This is exactly the V2.18 cold failure
signature.

To detect a missing patch from training metrics : if
`Loss/learning_rate` hits 1e-5 and stays there for > 50 iters, the
patch is missing or was reverted.

### Risk if applied

- Affects ALL rsl_rl-using configs in this venv, not just V218B. Old
  V2.x configs replayed for comparison will use the new floor too. This
  changes their behaviour subtly vs the historical runs that produced
  the metrics in `notes/eval2_pipeline.md`.
- Possible (low probability) that some other code path in rsl_rl or
  isaac_so_arm101 assumes LR can go below 1e-4. Not observed in
  practice.
- The patch is invisible to teammates who clone the repo and build
  their own venv. They need to run `patch_rsl_rl.py` once.

### How to verify the patch is applied

```powershell
Select-String -Path C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\.venv\Lib\site-packages\rsl_rl\algorithms\ppo.py `
  -Pattern "max\(1e-4"
```

Expected output : one line containing the patched expression.
