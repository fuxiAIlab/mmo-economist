import time
import os
import yaml
from os.path import dirname as fa_path


from ray.tune.logger import Logger

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Any, Dict
import numpy as np
class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        # info["episode"].user_data["res"] = []
        episode.user_data["res"] = []
        episode.user_data["rew_log"] = {str(i):[] for i in range(10)}
        # import ipdb;ipdb.set_trace()
        print("episode {} (env-idx={}) started.".format(
            episode.episode_id, env_index))
        # episode.user_data["pole_angles"] = []
        # episode.hist_data["pole_angles"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        if episode.last_info_for('0').get('res')[0]>0:
            episode.user_data["res"].append(episode.last_info_for('0')['res'])
        for agent_id in range(10):
            episode.user_data["rew_log"][str(agent_id)].append(episode.prev_reward_for(str(agent_id)))

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        res = np.array(episode.user_data["res"])
        idx = np.where(res[:, 0] > 0)[0]
        equality = np.split(res[:, 1], idx + 1)
        equality = np.mean([np.mean(_) for _ in equality if len(_) > 0])
        print(f'profit: {res[idx,0]}, equality: {equality}, capability: {res[idx,2]}')

        episode.custom_metrics["profit"] = np.mean(res[idx, 0])
        episode.custom_metrics["equality"] = equality
        episode.custom_metrics["capability"] = np.mean(res[idx, 2])

class MyPrintLogger(Logger):
    """Logs results by simply printing out everything.
    """

    def _init(self):
        # Custom init function.
        print("Initializing ...")
        # Setting up our log-line prefix.
        self.prefix = self.config.get("logger_config").get("prefix")

    def on_result(self, result: dict):
        print('fin')
        # print('rew:',result['policy_reward_mean'],result['episode_reward_mean'])
        # print('count:',result['custom_metrics']['capability_mean'],result['custom_metrics']['profit_mean'])
        # result['hist_stats']['policy_policy_p_reward']
        # result['hist_stats']['policy_policy_a_reward']

        # Define, what should happen on receiving a `result` (dict).
        # print(f"{self.prefix}: {result.keys()}")

    def close(self):
        # Releases all resources used by this logger.
        print("Closing")

    def flush(self):
        # Flushing all possible disk writes to permanent storage.
        print("Flushing ;)", flush=True)

def self_train(Trainer,config):
    trainer = Trainer(config)
    result = trainer.train()
    res=trainer.save('test')
    trainer.restore(res)
    import ipdb;ipdb.set_trace()
    NUM_ITER = 20
    cur_best = 0
    pst_time = time.time()

    trainer=Trainer(config)
    for iteration in range(NUM_ITER):
        print(f'********** Iter : {iteration} **********')
        result=trainer.train()

        cur_time = time.time()

        if 'a' in result['policy_reward_mean'].keys():
            if result['policy_reward_mean']['a'] > cur_best:
                cur_best = result['policy_reward_mean']['a']
                trainer.save(
                    f"dir_ckpt_{run_configuration['env']['adjustemt_type']}/{cfg_path[:9]}/rew_{round(cur_best, 4)}")
            iter_time = round(cur_time - pst_time, 4)
            episode_reward_mean = round(result.get('episode_reward_mean'), 4)
            a_rew = round(result['policy_reward_mean']['a'], 4)
            p_rew = round(result['policy_reward_mean']['p'], 4)

            if 'profit_mean' in result['custom_metrics'].keys():
                profit = round(result['custom_metrics']['profit_mean'], 4)
                equality = round(result['custom_metrics']['equality_mean'], 4)
                capability = round(result['custom_metrics']['capability_mean'], 4)
                print(f"time: {iter_time} epi_rew: {episode_reward_mean} a_rew:{a_rew} ",
                      f" p_rew:{p_rew}, epi_len: {result['episode_len_mean']}",
                      f" pro:{profit} equ:{equality} cap:{capability} ")
            pst_time = cur_time
        else:
            print(f"episode_reward_mean: {result.get('episode_reward_mean')}")
        if iteration % 20 == 19:
            trainer.save(f"./dir_ckpt_{run_configuration['env']['adjustemt_type']}/{cfg_path[:9]}/iter_{iteration}")
            print(f"save ckpt at iter {iteration}")

def get_original_cfg():
    print(os.path.abspath(__file__))
    repo_path = fa_path(fa_path(fa_path(fa_path(fa_path(__file__)))))
    cfg_path = os.path.join(repo_path, 'experiments', 'marl_config.yaml')
    run_configuration = yaml.safe_load(open(cfg_path))
    trainer_config = run_configuration.get("trainer")
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }
    trainer_config.update(
        {
            "env_config": env_config,
            "seed": 2014,
            "multiagent": config['multiagent'],
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
                                          * trainer_config.get("num_envs_per_worker"),
        }
    )