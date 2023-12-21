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


def generate_rollout_from_current_trainer_policy(
    trainer,
    env_obj,
    num_dense_logs=1
):
    dense_logs = {}

    for idx in range(num_dense_logs):
        planner_action = []
        planner_action_mask = []
        profitability = []
        equality = []

        # Set initial states
        agent_states = {}
        for agent_idx in range(env_obj.env.n_agents):
            agent_states[str(agent_idx)] = trainer.get_policy(
                "a").get_initial_state()
        planner_states = trainer.get_policy("p").get_initial_state()

        # Play out the episode
        obs = env_obj.reset(force_dense_logging=True)
        for t in range(env_obj.env.episode_length):
            actions = {}
            for agent_idx in range(env_obj.env.n_agents):
                # Use the trainer object directly to sample actions for each agent
                actions[str(agent_idx)] = trainer.compute_action(
                    obs[str(agent_idx)],
                    agent_states[str(agent_idx)],
                    policy_id="a",
                    full_fetch=False
                )

            # Action sampling for the planner
            actions["p"] = trainer.compute_action(
                obs['p'],
                planner_states,
                policy_id='p',
                full_fetch=False
            )
            actions = {k: v[0] #if k != 'p' else [v[0]]
                       for k, v in actions.items()}

            planner_action.append(actions['p'])
            planner_action_mask.append(obs['p']['action_mask'])
            metric=env_obj.env.scenario_metrics()
            profitability.append(metric['social/profitability'])
            equality.append(metric['social/equality'])

            obs, rew, done, info = env_obj.step(actions)
            
            if done['__all__']:
                break
        dense_logs[idx] = env_obj.env.dense_log
        dense_logs[idx]['planner_action'] = planner_action
        dense_logs[idx]['profitability'] = profitability
        dense_logs[idx]['equality'] = equality

        dense_logs[idx]['mon_sense'] =env_obj.env._player_monetary_cost_sensitivities
        dense_logs[idx]['nonmon_sense']= env_obj.env._player_nonmonetary_cost_sensitivities
    return dense_logs



ray.init(webui_host="127.0.0.1")


# tmp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--adj', type=str, default='planner')
parser.add_argument('--restore', type=str, default='')
parser.add_argument('--cfg', type=str, default='')
res =parser.parse_args()

config_path = os.path.join('./experiments', res.cfg)
# config_path = os.path.join('./experiments', "config_80_20.yaml")

with open(config_path, "r") as f:
    run_configuration = yaml.safe_load(f)

trainer_config = run_configuration.get("trainer")
run_configuration["env"]["adjustemt_type"]=res.adj

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
# trainer._restore('ckpt/train_actor/rew_39.4727/checkpoint_193/checkpoint-193')

if res.restore!='':
    run_configuration["general"]["restore"]=res.restore
if 'restore' in run_configuration['general'].keys():
    trainer._restore(run_configuration['general']['restore'])
    print(f"restore {run_configuration['general']['restore']} , {res.adj}")

dense_logs = generate_rollout_from_current_trainer_policy(
    trainer,
    dummy_env,
    num_dense_logs=1
)

print('monetary sensetivities:',dummy_env.env._player_monetary_cost_sensitivities)
print('nonmonetary sensetivities:',dummy_env.env._player_nonmonetary_cost_sensitivities)

# test_config='20'
p1,p2,p3,p4=plotting.breakdown(dense_logs[0])
p1.savefig(f'p1_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
p2.savefig(f'p2_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
p3.savefig(f'p3_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
p4.savefig(f'p4_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
np.save(f'{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}_dense_logs.npy',dense_logs)

