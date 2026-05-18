# Codebase guide — full env / training / MDP / DR source

`squint-native-iso-src.zip` = the **complete** env + training repo (code
only; `runs/`, `.git`, logs, checkpoints stripped). Teammates get exactly
what was used — full ability to add DR, add cubes, change the MDP, retrain.

```bash
unzip squint-native-iso-src.zip && cd squint-native-iso
conda env create -f environment.yaml   # creates conda env 'squint'
conda activate squint
```

## Where everything lives

| File | What |
|---|---|
| `train_squint.py` | SAC+C51 trainer. All CLI flags (tyro): `--env-id --image-size --render-size --num-envs --buffer-size --total-timesteps --num-updates --checkpoint(warm) --exp-name --env-domain-randomization/--no- --apply-jitter/--no- --exposure-dr/--no- --dr-config-json --evaluate`. ckpt_best on `success_at_end` → `runs/<exp>/ckpt_best.pt`. |
| `envs/place.py` | **The MDP.** PlaceCube envs + cube/bowl scene, `_get_obs`, reward, `evaluate()`/success, terminations. Registered ids: `SO101PlaceCube-v1` (Eval1), `SO101PlaceCubeEval2-v1` (colour-cond), `SO101PlaceCubeEval3-v1` (multi-cube/seq), `SO101PlaceCan-v1`. |
| `envs/base_random_env.py` | **All DR knobs.** `RandomizationConfig` dataclass: gripper/arm stiffness·damping, `action_delay_steps_range`, `lag_alpha_range`, `randomize_lighting`, `ambient_range`, `directional_intensity_range`, `robot_color`, wrist/third camera noise. Merged at runtime via `--dr-config-json`. |
| `utils.py` | Obs/DR wrappers: `ExposureDRWrapper` (now with `EXPO_DR_SCALE` env var, reduced envelope), `ColorJitterWrapper`, `DownsampleObsWrapper`. |
| `dr_configs/` | Example DR config JSONs (graded curricula). |
| `view_policy.py`,`view_env.py`,`examples/visualize_sim.py` | Visualize env / a policy (GUI or 50-demo planche). |
| `deploy_utils/manipulator.py`, `deploy.py` | Real-robot bridge — carries the validated real↔sim mapping (see `MAPPING.md`). `.bak.*` files are just backups. |
| `environment.yaml` | Conda env spec. |

## How-to

**Add / tune DR (no code):** write a JSON overriding `RandomizationConfig`
fields, pass `--dr-config-json yours.json` (+ `--env-domain-randomization`).
Examples in `dr_configs/` and `../curriculum/dr_*.json`. This is a sim2real
knob only — does **not** touch reward/evaluate (keeps the env NATIVE).

**Add a new DR type (code):** add a field to `RandomizationConfig`
(`envs/base_random_env.py`) and apply it in that file's randomization hooks
(see how `arm_stiffness_range`/`ambient_range` are consumed).

**Exposure DR severity:** `EXPO_DR_SCALE=<0..1> python train_squint.py …
--exposure-dr` — scales the (reduced) envelope from neutral→max. Used by the
Eval2 fine curriculum (0.30/0.60/1.00). Envelope defaults in
`utils.ExposureDRWrapper.__init__`.

**Add 6 cubes / multi-cube:** `SO101PlaceCubeEval3-v1` in `envs/place.py` is
already a multi-cube / multi-goal env — copy its pattern (cube list, per-cube
spawn, goal sequence) or parametrize the cube count in the PlaceCube base.
The 6 cube colours are handled in `place.py` (goal-colour one-hot for Eval2).

**Change the MDP (success / reward):** edit `evaluate()` / reward in
`envs/place.py`. ⚠️ This is exactly the documented genuine-ceiling fix
(making "success" require a real release / strict in-bowl check). It breaks
the "keep-native" rule by design — a deliberate team decision (see
`WORKLOG_AND_TASKS.md` open item #1).

**Train / warm-start / curriculum:** `train_squint.py` + the orchestrators in
`../curriculum/` (`_curriculum_run.sh`, `_eval2_fine.sh`). OOM-safe params:
`--num-envs 512 --buffer-size 200000 --image-size 64 --render-size 128`.
`../curriculum/CURRICULUM_STATE.md` = exact log of the run that produced the
shipped checkpoints.

**Visualize:** `python view_policy.py --ckpt <ckpt_best.pt> --demos 50
--render rgb_array --env-id SO101PlaceCube-v1 --obs-mode rgb --image-size 64
--save` (planche to ~/Desktop) or `--render human` for an interactive GUI.
`python examples/visualize_sim.py` to just look at the env.

## Note on credentials
The codebase was scanned — **no API keys / tokens / passwords** are present
and **none are shipped**. Set your own robot setup values (`ROBOT_PORT`,
`CAMERA_INDEX`, `CALIBRATION_ID`, `CALIBRATION_DIR`) at the top of the infer
scripts / `deploy_utils/robot_config.py`. W&B is opt-in (`--track`, off by
default; supply your own entity).
