# -*- coding: utf-8 -*-
"""M1 (reviewer 2): retrain [98]'s released XRouting on the OPEN-boundary network with the
released reward (compute_reward = -travel_time) and released architecture. The 2022 env
restarts a fresh SUMO every episode and leaks until a native access violation after ~a dozen
resets, so training is CHUNKED: each launch restores the last checkpoint, runs XR_CHUNK iters
(staying under the crash threshold), saves, and exits. An external loop relaunches until
XR_ITERS is reached. Only the reward/model-independent knobs (batch, workers, chunking)
differ from rl/model_config.XRouting_config. Checkpoints -> experiments/xrouting_mc4/ckpt_open.
Exit 42 = all iters done; 0 = chunk done, relaunch; other = failure."""
import sys, os, time, json
XRO = os.environ.get("XR_WORKDIR", r"C:\xro")
OBS = int(os.environ.get("XR_OBS", "46"))
sys.path.insert(0, XRO); os.chdir(XRO)
import ray
from ray.rllib.agents.ppo import PPOTrainer
from rl.env.dynamic_rerouting_env import DynamicRerouteEnv
from utils.registry import create_env
from ray.tune.registry import register_env
from rl.model_config import ModelConfig

OUT = os.environ.get("XR_OUT", r"D:\review_paper\drl-rgs-review\experiments\xrouting_mc4\ckpt_open")
os.makedirs(OUT, exist_ok=True)
STATE = os.path.join(OUT, "state.json")
LOG = os.path.join(OUT, "train_log.json")
STOP_ITERS = int(os.environ.get("XR_ITERS", "50"))
CHUNK = int(os.environ.get("XR_CHUNK", "3"))
BATCH = int(os.environ.get("XR_BATCH", "384"))

state = {"done_iters": 0, "last_ckpt": None}
if os.path.exists(STATE):
    with open(STATE) as f:
        state = json.load(f)
log = []
if os.path.exists(LOG):
    with open(LOG) as f:
        log = json.load(f)

if state["done_iters"] >= STOP_ITERS:
    print("ALREADY COMPLETE at iter %d" % state["done_iters"], flush=True)
    sys.exit(42)

env, env_name = create_env(params=dict(env_name=DynamicRerouteEnv, version=0,
    reward_threshold=-200, max_episode_steps=1000, observation_size=OBS, action_size=4,
    initial_edge="right0D0", destination="A2left2", work_dir=XRO, model="XRouting"))
register_env(env_name, env)
configuration = ModelConfig(num_gpus=0, num_cpus=1, num_workers=6, num_cpus_per_worker=1)
config = configuration.XRouting_config(env_name=env_name)
config["num_workers"] = 0
config["num_gpus"] = 0
config["train_batch_size"] = BATCH
config["sgd_minibatch_size"] = min(256, BATCH)
config["seed"] = int(os.environ.get("XR_SEED", "0"))

ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False,
         log_to_driver=False)
trainer = PPOTrainer(config=config)
if state["last_ckpt"] and os.path.exists(state["last_ckpt"]):
    trainer.restore(state["last_ckpt"])
    print("RESTORED from %s (iter %d)" % (state["last_ckpt"], state["done_iters"]), flush=True)

t0 = time.time()
target = min(state["done_iters"] + CHUNK, STOP_ITERS)
for it in range(state["done_iters"] + 1, target + 1):
    r = trainer.train()
    row = dict(iter=it, reward_mean=r.get("episode_reward_mean"),
               reward_max=r.get("episode_reward_max"),
               len_mean=r.get("episode_len_mean"), ts=round(time.time() - t0, 1))
    log.append(row)
    print("ITER %d reward_mean=%s len_mean=%s" % (it, row["reward_mean"], row["len_mean"]),
          flush=True)

ckpt = trainer.save(OUT)
state = {"done_iters": target, "last_ckpt": ckpt}
with open(STATE, "w") as f:
    json.dump(state, f, indent=2)
with open(LOG, "w") as f:
    json.dump(log, f, indent=2)
print("CHUNK DONE -> iter %d, ckpt=%s" % (target, ckpt), flush=True)
ray.shutdown()
sys.exit(42 if target >= STOP_ITERS else 0)
