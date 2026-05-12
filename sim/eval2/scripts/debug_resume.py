"""Diagnose why resume is not being applied."""
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import sim.eval2  # noqa
from sim.eval2.agents.rsl_rl_ppo_cfg_v2_16_resume import LiftCubePPORunnerCfgV216Resume

print("=" * 60)
print("STEP 1 - Instantiate the class directly")
inst = LiftCubePPORunnerCfgV216Resume()
print(f"  inst.resume          = {inst.resume!r}")
print(f"  inst.load_run        = {inst.load_run!r}")
print(f"  inst.load_checkpoint = {inst.load_checkpoint!r}")
print(f"  type(inst.resume)    = {type(inst.resume)}")

print()
print("=" * 60)
print("STEP 2 - Check via to_dict()")
d = inst.to_dict()
print(f"  d['resume']          = {d.get('resume')!r}")
print(f"  d['load_run']        = {d.get('load_run')!r}")
print(f"  d['load_checkpoint'] = {d.get('load_checkpoint')!r}")

print()
print("=" * 60)
print("STEP 3 - Check via __dict__")
print(f"  __dict__ keys with 'resume'/'load': "
      f"{[k for k in inst.__dict__ if 'resume' in k or 'load' in k]}")
for k in ['resume', 'load_run', 'load_checkpoint']:
    print(f"  inst.__dict__.get('{k}') = {inst.__dict__.get(k)!r}")

print()
print("=" * 60)
print("STEP 4 - Check dataclass fields")
import dataclasses
for f in dataclasses.fields(inst):
    if f.name in ('resume', 'load_run', 'load_checkpoint'):
        print(f"  field name={f.name}  default={f.default!r}  type={f.type}")

print()
print("=" * 60)
print("STEP 5 - register_task_to_hydra simulation")
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
cfg = load_cfg_from_registry("Isaac-LeIsaac-SO101-Lift-Visual-V216-v0", "rsl_rl_cfg_entry_point")
print(f"  cfg.resume          = {cfg.resume!r}")
print(f"  cfg.load_run        = {cfg.load_run!r}")
print(f"  cfg.load_checkpoint = {cfg.load_checkpoint!r}")
d = cfg.to_dict()
print(f"  cfg.to_dict()['resume'] = {d.get('resume')!r}")

print()
print("=" * 60)
print("STEP 6 - Full register_task_to_hydra + Hydra simulation")
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

env_cfg, agent_cfg = register_task_to_hydra(
    "Isaac-LeIsaac-SO101-Lift-Visual-V216-v0".split(":")[-1],
    "rsl_rl_cfg_entry_point"
)
print(f"After register_task_to_hydra:")
print(f"  agent_cfg.resume      = {agent_cfg.resume!r}")
print(f"  agent_cfg.load_run    = {agent_cfg.load_run!r}")
print(f"  agent_cfg.to_dict()['resume'] = {agent_cfg.to_dict().get('resume')!r}")

# Retrieve what's stored in Hydra ConfigStore
print()
print("ConfigStore content:")
cs = ConfigStore.instance()
node = cs.repo.get("Isaac-LeIsaac-SO101-Lift-Visual-V216-v0.yaml")
if node is not None:
    stored = node.node
    print(f"  stored type : {type(stored)}")
    if isinstance(stored, dict):
        agent_part = stored.get('agent', {})
        print(f"  stored['agent']['resume'] = {agent_part.get('resume')!r}")
        print(f"  stored['agent']['load_run'] = {agent_part.get('load_run')!r}")
    else:
        # OmegaConf node
        as_dict = OmegaConf.to_container(stored, resolve=True)
        print(f"  stored agent resume = {as_dict.get('agent', {}).get('resume')!r}")
        print(f"  stored agent load_run = {as_dict.get('agent', {}).get('load_run')!r}")
else:
    print(f"  node not found by that key, listing all keys:")
    for k in cs.repo.keys():
        print(f"    {k}")

# Simulate the from_dict round-trip with the same agent dict
print()
print("STEP 7 - simulate agent_cfg.from_dict() with the SAME dict")
print(f"  BEFORE: agent_cfg.resume = {agent_cfg.resume!r}")
test_dict = agent_cfg.to_dict()
print(f"  test_dict['resume'] = {test_dict['resume']!r}")
agent_cfg.from_dict(test_dict)
print(f"  AFTER (round-trip same dict): agent_cfg.resume = {agent_cfg.resume!r}")

# Simulate with an explicit False
print()
print("STEP 8 - simulate from_dict with resume=False (should it stick?)")
agent_cfg.resume = True  # reset
print(f"  BEFORE: agent_cfg.resume = {agent_cfg.resume!r}")
agent_cfg.from_dict({"resume": False})
print(f"  AFTER from_dict({{'resume': False}}): agent_cfg.resume = {agent_cfg.resume!r}")

app.close()
