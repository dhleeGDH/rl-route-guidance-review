# -*- coding: utf-8 -*-
"""M1 completion measurement, corrected. The env resolves destination arrival through SUMO's
own arrival (not a traci.vehicle.remove call), so completion must be read from SUMO's tripinfo
output, where each rl vehicle records its arrivalLane. Roll out a restored checkpoint on the
released single-agent env for a few episodes, then parse the per-episode tripinfo:
    completed  = rl vehicles whose arrivalLane is on the destination edge (A2left2)
    left       = rl vehicles that arrived on some other edge (leaked out / wrong sink)
    unfinished = rl vehicles still en route when the episode ended
    completion rate = completed / total rl vehicles.
Env/reward/model are the released ones; only batch/workers/chunking differed in training."""
import sys, os, json, glob
import xml.etree.ElementTree as ET
XRO = os.environ.get("XR_WORKDIR", r"C:\xro")
OBS = int(os.environ.get("XR_OBS", "46"))
CKPT = os.environ["XR_CKPT"]
EPISODES = int(os.environ.get("XR_EVAL_EPISODES", "8"))
DEST = "A2left2"
sys.path.insert(0, XRO); os.chdir(XRO)
import ray
from ray.rllib.agents.ppo import PPOTrainer
from rl.env.dynamic_rerouting_env import DynamicRerouteEnv
from utils.registry import create_env
from ray.tune.registry import register_env
from rl.model_config import ModelConfig
import traci

TRIP_DIR = os.path.join(XRO, "training_tripinfo", "XRouting_training")
# clear stale tripinfo so we only parse this run's episodes
for f in glob.glob(os.path.join(TRIP_DIR, "tripinfo*.xml")):
    try: os.remove(f)
    except Exception: pass

env_creator, env_name = create_env(params=dict(env_name=DynamicRerouteEnv, version=0,
    reward_threshold=-200, max_episode_steps=1000, observation_size=OBS, action_size=4,
    initial_edge="right0D0", destination=DEST, work_dir=XRO, model="XRouting"))
register_env(env_name, env_creator)
env = DynamicRerouteEnv(observation_size=OBS, action_size=4, initial_edge="right0D0",
                        destination=DEST, work_dir=XRO, model="XRouting", nogui=True)
configuration = ModelConfig(num_gpus=0, num_cpus=1, num_workers=6, num_cpus_per_worker=1)
config = configuration.XRouting_config(env_name=env_name)
config["num_workers"] = 0; config["num_gpus"] = 0
config["train_batch_size"] = 320; config["sgd_minibatch_size"] = 256
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
trainer = PPOTrainer(config=config)
trainer.restore(CKPT)
try:
    traci.close(False)
except Exception:
    pass

import numpy as _np
RANDOM = os.environ.get("XR_RANDOM", "0") == "1"
for ep in range(EPISODES):
    obs = env.reset(); done = False; steps = 0
    while not done and steps < 400:
        if RANDOM:
            m = _np.asarray(obs["action_mask"]); valid = _np.where(m > 0)[0]
            a = int(_np.random.choice(valid)) if len(valid) else 0
        else:
            a = trainer.compute_single_action(obs, explore=False)
        obs, r, done, info = env.step(a); steps += 1
    print("ep %d rollout done (steps=%d)" % (ep, steps), flush=True)
# close final SUMO so the last tripinfo flushes
try:
    traci.close(False)
except Exception:
    pass
ray.shutdown()

def parse(path):
    comp = left = unfin = 0
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return 0, 0, 0
    for t in root.findall("tripinfo"):
        if not t.get("id", "").startswith("rl"):
            continue
        al = t.get("arrivalLane", ""); ar = t.get("arrival", "-1")
        if al.startswith(DEST + "_") or al == DEST:
            comp += 1
        elif ar in ("-1", "") or al == "":
            unfin += 1
        else:
            left += 1
    return comp, left, unfin

C = L = U = 0
files = sorted(glob.glob(os.path.join(TRIP_DIR, "tripinfo*.xml")))
for f in files:
    c, l, u = parse(f); C += c; L += l; U += u
total = C + L + U
rate = (C / total * 100.0) if total else float("nan")
out = {"workdir": XRO, "obs": OBS, "ckpt": CKPT, "episodes": EPISODES,
       "tripinfo_files": len(files), "total_rl": total, "completed": C,
       "left_boundary_or_wrong": L, "unfinished": U,
       "completion_rate_pct": round(rate, 1)}
print("RESULT " + json.dumps(out), flush=True)
with open(os.path.join(os.path.dirname(CKPT), "..", "eval_completion2.json"), "w") as f:
    json.dump(out, f, indent=2)
