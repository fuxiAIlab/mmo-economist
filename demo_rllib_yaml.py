import yaml
import os
import foundation
import numpy as np
import matplotlib.pyplot as plt
import ray

from experiments import tf_models
from foundation.utils import plotting
from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.algorithms.ppo import PPO


 


ray.init(webui_host='127.0.0.1')

config_path = os.path.join('./experiments', "config_50_50.yaml")
# config_path = os.path.join('./experiments', "config_80_20.yaml")

with open(config_path, "r") as f:
    run_configuration = yaml.safe_load(f)

trainer_config = run_configuration.get("trainer")
env_config = {
    "env_config_dict": run_configuration.get("env"),
    "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
}

dummy_env = RLlibEnvWrapper(env_config, verbose=True)

agent_policy_tuple = (
    None,
    dummy_env.observation_space,
    dummy_env.action_space,
    run_configuration.get("agent_policy"),
)
planner_policy_tuple = (
    None,
    dummy_env.observation_space_pl,
    dummy_env.action_space_pl,
    run_configuration.get("planner_policy"),
)

policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}
def policy_mapping_fun(i): return "a" if str(i).isdigit() else "p"


if run_configuration["general"]["train_planner"]:
    policies_to_train = ["a", "p"]
else:
    policies_to_train = ["a"]

trainer_config.update(
    {
        "env_config": env_config,
        "seed": 2014,
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fun,
        },
        "metrics_smoothing_episodes": trainer_config.get("num_workers")
        * trainer_config.get("num_envs_per_worker"),
    }
)
trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)

NUM_ITERS = 300
cur_best=0
import time
pst_time=time.time()
for iteration in range(NUM_ITERS):
    print(f'********** Iter : {iteration} **********')
    result = trainer.train()

    #r1 = trainer.workers.local_worker().sampler.get_data().policy_batches
    #print(r1['a']['rewards'].sum(), result['episode_reward_mean'])

    cur_time = time.time()

    if 'p' in result['policy_reward_mean'].keys():
        if result['policy_reward_mean']['p']>cur_best:
        # if result.get('episode_reward_mean')>cur_best:
            cur_best= result['policy_reward_mean']['p'] #result.get('episode_reward_mean')
            trainer.save(f'./ckpt_planner_50/rew_{round(cur_best,4)}')
        iter_time=round(cur_time-pst_time,4)
        episode_reward_mean=round(result.get('episode_reward_mean'),4)
        a_rew=round(result['policy_reward_mean']['a'],4)
        p_rew=round(result['policy_reward_mean']['p'],4)

        print(f"time: {iter_time} episode_reward_mean: {episode_reward_mean} a_rew:{a_rew} ",
              f" p_rew:{p_rew}, epsidoe_length: {result['episode_len_mean'] }")
        pst_time=cur_time
    else:
        print(f"episode_reward_mean: {result.get('episode_reward_mean')}")
trainer.save(f'./ckpt_planner_50/last_ckpt')
