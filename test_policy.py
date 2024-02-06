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

config_path = os.path.join('./experiments', "config_50_50.yaml")

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
# trainer._restore('ckpts/dir_ckpt_random-asy/t1/iter_399/checkpoint_400/checkpoint-400')
trainer._restore('ckpts/dir_ckpt_random-asy/iter_60/checkpoint_61/checkpoint-61')

# dense_logs = generate_rollout_from_current_trainer_policy(
#     trainer,
#     dummy_env,
#     num_dense_logs=3
# )

dense_logs=[]
for _  in range(15):
    # trainer._restore(res.restore)
    # trainer._restore('dir_ckpt_random-asy/iter_399/checkpoint_400/checkpoint-400')
    # trainer._restore('ckpts/dir_ckpt_tune_random-asy/iter_399/checkpoint_400/checkpoint-400')
    # trainer._restore('ckpts/dir_ckpt_random-asy/iter_199/checkpoint_200/checkpoint-200')

    res=generate_rollout_from_current_trainer_policy(trainer,dummy_env,1)
    recharge = np.sum([len(res[0]['Recharge'][j]) for j in range(len(res[0]['Upgrade']))])
    upgrade = np.sum([len(res[0]['Upgrade'][j]) for j in range(len(res[0]['Upgrade']))])
    print(f'test: Recharge: {recharge} Upgrade: {upgrade} epi_len: {len(res[0]["Upgrade"])}')

    # trainer._restore('/home/game/used/used/mmo-economist/runs/ckpts/t1/ckpt_50/last_ckpt/checkpoint_500/checkpoint-500')
    # dummy_env = RLlibEnvWrapper(env_config, verbose=False)
    # tmp=generate_rollout_from_current_trainer_policy(trainer,dd,1)
    # dense_logs.append(tmp[0])
print('monetary sensetivities:',dummy_env.env._player_monetary_cost_sensitivities)
print('nonmonetary sensetivities:',dummy_env.env._player_nonmonetary_cost_sensitivities)

# test_config='20'
p1,p2,p3,p4=plotting.breakdown(dense_logs[0])
p1.savefig(f'p1.png')
p2.savefig(f'p2.png')
p3.savefig(f'p3.png')
p4.savefig(f'p4.png')
np.save(f'dense_logs.npy',dense_logs)

# p1.savefig(f'p1_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
# p2.savefig(f'p2_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
# p3.savefig(f'p3_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
# p4.savefig(f'p4_{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}.png')
# np.save(f'{res.cfg[:10]}_{env_config["env_config_dict"]["adjustemt_type"]}_dense_logs.npy',dense_logs)

