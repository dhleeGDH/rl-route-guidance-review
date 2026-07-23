# -*- coding: utf-8 -*-
"""M1 completion measurement for [98]'s released XRouting. Restore a trained checkpoint and
run the released single-agent decision-zone env, which spawns rl vehicles one after another
into the traffic stream. The env calls traci.vehicle.remove(reason=2) ONLY when a vehicle
reaches the destination edge (dynamic_rerouting_env.py L241-242); vehicles that leak out the
open boundary are auto-removed by SUMO with no such call. So, over the rl vehicles spawned in
a few episodes (kept under the ~12-episode SUMO-restart crash threshold):
    trip completion rate = (destination removals) / (rl_car spawns).
Env/reward/model are the released ones; only batch/workers/chunking differed in training.
Run AFTER the training harness finishes (the orchestrator kills stray python/SUMO)."""
import sys, os, json
XRO = os.environ.get("XR_WORKDIR", r"C:\xro")
OBS = int(os.environ.get("XR_OBS", "46"))
CKPT = os.environ["XR_CKPT"]
EPISODES = int(os.environ.get("XR_EVAL_EPISODES", "8"))
sys.path.insert(0, XRO); os.chdir(XRO)
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from rl.env.dynamic_rerouting_env import DynamicRerouteEnv
from utils.registry import create_env
from ray.tune.registry import register_env
from rl.model_config import ModelConfig
import traci

# ---- instrument the two traci calls that define completion ----
_stats = {"spawned": 0, "completed": 0, "dest_edges": {}}
_add, _remove = traci.vehicle.add, traci.vehicle.remove
def add_wrap(*a, **k):
    if k.get("typeID") == "rl_car" or (len(a) >= 3 and a[2] == "rl_car"):
        _stats["spawned"] += 1
    return _add(*a, **k)
def remove_wrap(vehID, *a, **k):
    try:
        edge = traci.vehicle.getRoadID(vehID)
        _stats["dest_edges"][edge] = _stats["dest_edges"].get(edge, 0) + 1
    except Exception:
        pass
    _stats["completed"] += 1
    return _remove(vehID, *a, **k)
traci.vehicle.add = add_wrap
traci.vehicle.remove = remove_wrap

# register the trainer's env name (needed by the restored config), but roll out on a
# directly-instantiated env (create_env returns a creator fn + name, not an instance).
env_creator, env_name = create_env(params=dict(env_name=DynamicRerouteEnv, version=0,
    reward_threshold=-200, max_episode_steps=1000, observation_size=OBS, action_size=4,
    initial_edge="right0D0", destination="A2left2", work_dir=XRO, model="XRouting"))
register_env(env_name, env_creator)
env = DynamicRerouteEnv(observation_size=OBS, action_size=4, initial_edge="right0D0",
                        destination="A2left2", work_dir=XRO, model="XRouting", nogui=True)
configuration = ModelConfig(num_gpus=0, num_cpus=1, num_workers=6, num_cpus_per_worker=1)
config = configuration.XRouting_config(env_name=env_name)
config["num_workers"] = 0; config["num_gpus"] = 0
config["train_batch_size"] = 320; config["sgd_minibatch_size"] = 256
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
trainer = PPOTrainer(config=config)
trainer.restore(CKPT)
# PPOTrainer builds its own env (num_workers=0 local worker), which seizes the single traci
# 'default' connection. compute_single_action needs only the policy, so release it before we
# roll out on our instrumented env.
try:
    traci.close(False)
except Exception:
    pass

for ep in range(EPISODES):
    obs = env.reset(); done = False; steps = 0
    while not done and steps < 300:
        a = trainer.compute_single_action(obs, explore=False)
        obs, r, done, info = env.step(a); steps += 1
    print("ep %d done spawned=%d completed=%d" % (ep, _stats["spawned"], _stats["completed"]),
          flush=True)

rate = (_stats["completed"] / _stats["spawned"] * 100.0) if _stats["spawned"] else float("nan")
out = {"workdir": XRO, "obs": OBS, "ckpt": CKPT, "episodes": EPISODES,
       "spawned": _stats["spawned"], "completed": _stats["completed"],
       "completion_rate_pct": round(rate, 1), "terminal_edges": _stats["dest_edges"]}
print("RESULT " + json.dumps(out), flush=True)
with open(os.path.join(os.path.dirname(CKPT), "..", "eval_completion.json"), "w") as f:
    json.dump(out, f, indent=2)
ray.shutdown()
