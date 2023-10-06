import yaml
import os
import foundation
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray import tune

from experiments import tf_models
from foundation.utils import plotting
from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.algorithms.ppo import PPO


ray.init(webui_host='127.0.0.1')


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--adj', type=str, default='planner')
# parser.add_argument('--restore', type=str, default='')
# parser.add_argument('--cfg', type=str, default='')
# res =parser.parse_args()
# cfg_path=res.cfg



#config_path = os.path.join('./experiments', "config_50_50.yaml")
cfg_path='config_20_80.yaml'
config_path = os.path.join('./experiments', cfg_path)

with open(config_path, "r") as f:
    run_configuration = yaml.safe_load(f)

trainer_config = run_configuration.get("trainer")

def on_episode_start(info):
    info["episode"].user_data["res"] = []
    # info["episode"].hist_data["res"] = []


def on_episode_step(info):
    # print(info['episode'].total_reward)

    if info['episode'].last_info_for('p').get('res') is not None:
        # if info['episode'].last_info_for('p').get('res')[0]>0:
        info["episode"].user_data["res"].append(info['episode'].last_info_for('p')['res'])



def on_episode_end(info):
    episode = info["episode"]
    res=np.array(episode.user_data["res"])
    idx=np.where(res[:,0]>0)[0]
    equality=np.split(res[:, 1], idx + 1)
    equality=np.mean([np.mean(_) for _ in equality if len(_)>0])
    # print(f'profit: {res[idx,0]}, equality: {equality}, capability: {res[idx,2]}')

    info["episode"].custom_metrics["profit"] = np.mean(res[idx,0])
    info["episode"].custom_metrics["equality"] = equality
    info["episode"].custom_metrics["capability"] = np.mean(res[idx,2])
    # info["episode"].hist_data["res"] = np.mean(episode.user_data["res"])


# run_configuration["env"]["adjustemt_type"]=res.adj

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

import ipdb;ipdb.set_trace()
trainer_config['callbacks']={
    "on_episode_start": on_episode_start,
    "on_episode_step": on_episode_step,
    "on_episode_end": on_episode_end,
    # "on_sample_end": on_sample_end,
    # "on_postprocess_traj": on_postprocess_traj
    # "on_train_result": on_train_result,
                             }

trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)

# trainer._restore(res.restore)
# print(f'train {res.cfg}, load {res.restore}')
# trainer._restore('dir_ckpt_random-asy/iter_399/checkpoint_400/checkpoint-400')
NUM_ITERS = 400
cur_best=0
import time
pst_time=time.time()

for iteration in range(NUM_ITERS):
    print(f'********** Iter : {iteration} **********')
    result = trainer.train()
    # import ipdb;ipdb.set_trace()
    #r1 = trainer.workers.local_worker().sampler.get_data().policy_batches
    #print(r1['a']['rewards'].sum(), result['episode_reward_mean'])

    cur_time = time.time()

    if 'a' in result['policy_reward_mean'].keys():
        if result['policy_reward_mean']['a']>cur_best:
            cur_best=result['policy_reward_mean']['a']
            trainer.save(f"dir_ckpt_{run_configuration['env']['adjustemt_type']}/{cfg_path[:9]}/rew_{round(cur_best,4)}")
        iter_time=round(cur_time-pst_time,4)
        episode_reward_mean=round(result.get('episode_reward_mean'),4)
        a_rew=round(result['policy_reward_mean']['a'],4)
        p_rew=round(result['policy_reward_mean']['p'],4)

        if 'profit_mean' in result['custom_metrics'].keys():
            profit=round(result['custom_metrics']['profit_mean'],4)
            equality=round(result['custom_metrics']['equality_mean'],4)
            capability=round(result['custom_metrics']['capability_mean'],4)
            print(f"time: {iter_time} epi_rew: {episode_reward_mean} a_rew:{a_rew} ",
                  f" p_rew:{p_rew}, epi_len: {result['episode_len_mean'] }",
                  f" pro:{profit} equ:{equality} cap:{capability} ")
        pst_time=cur_time
    else:
        print(f"episode_reward_mean: {result.get('episode_reward_mean')}")
    if iteration  % 20 == 19 :
        trainer.save(f"./dir_ckpt_{run_configuration['env']['adjustemt_type']}/{cfg_path[:9]}/iter_{iteration}")
        print(f"save ckpt at iter {iteration}")


# trainer.save(f"./ckpt_{cfg_path[7:9]}/last_ckpt")
# trainer.save(f"test_exp/2trainer/ckpt_{cfg_path[7:9]}/last_ckpt")
