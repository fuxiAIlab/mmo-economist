from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict

import numpy as np

class TrainerCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.user_data["res"] = []
        # info["episode"].hist_data["res"] = []

    def on_episode_step(self,*, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # print(info['episode'].total_reward)
        if episode.last_info_for('group_all_')['_group_info'][0].get('res') is not None:
            episode.user_data["res"].append(episode.last_info_for('group_all_')['_group_info'][0]['res'])


        # if episode.last_info_for('0').get('res') is not None:
        #     # if info['episode'].last_info_for('0').get('res')[0]>0:
        #     episode.user_data["res"].append(episode.last_info_for('0')['res'])

    def on_episode_end(self,*, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        res = np.array(episode.user_data["res"])
        idx = np.where(res[:, 0] > 0)[0]
        equality = np.split(res[:, 1], idx + 1)
        equality = np.mean([np.mean(_) for _ in equality if len(_) > 0])
        # print(f'profit: {res[idx,0]}, equality: {equality}, capability: {res[idx,2]}')

        episode.custom_metrics["profit"] = np.mean(res[idx, 0])
        episode.custom_metrics["equality"] = equality
        episode.custom_metrics["capability"] = np.mean(res[idx, 2])
        # info["episode"].hist_data["res"] = np.mean(episode.user_data["res"])

