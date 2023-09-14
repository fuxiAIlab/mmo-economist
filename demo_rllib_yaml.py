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

config_path = os.path.join('./experiments', "config_80_20.yaml")

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
#trainer._restore('tmp_3.4641548739493224/checkpoint_9/checkpoint-9')
#trainer._restore('ckpt_a/tmp_37.74633408773801/checkpoint_158/checkpoint-158')
#trainer._restore('ckpt_a/tmp_39.47267792436246/checkpoint_193/checkpoint-193')
#trainer._restore('tmp_6.800040908919548/checkpoint_1049/checkpoint-1049')
NUM_ITERS = 1500
cur_best=0
for iteration in range(NUM_ITERS):
    print(f'********** Iter : {iteration} **********')
    result = trainer.train()
    if 'p' in result['policy_reward_mean'].keys():
        if result['policy_reward_mean']['p']>cur_best:
            # if result.get('episode_reward_mean')>cur_best:
            cur_best= result['policy_reward_mean']['p'] #result.get('episode_reward_mean')
            trainer.save(f'./ckpt_mask/rew_{round(cur_best,4)}')
        print(f"episode_reward_mean: {result.get('episode_reward_mean')}, "
              f"a_rew:{result['policy_reward_mean']['a']} ",
              f" p_rew:{result['policy_reward_mean']['p']}")
    else:
        print(f"episode_reward_mean: {result.get('episode_reward_mean')}")
