# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray import tune
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as JointQ_Config
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.core.VD.iql_vdn_qmix import JointQTrainer
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.scripts.coma import restore_model
import json
from typing import Any, Dict
from ray.tune.analysis import ExperimentAnalysis

import numpy as np
def run_joint_q(model: Any, exp: Dict, run: Dict, env: Dict,
                stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the IQL, VDN, and QMIX algorithm using Ray RLlib.
    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """

    ModelCatalog.register_custom_model(
        "Joint_Q_Model", model)

    _param = AlgVar(exp)

    algorithm = exp["algorithm"]
    episode_limit = env["episode_limit"]
    train_batch_episode = _param["batch_episode"]
    lr = _param["lr"]
    buffer_size = _param["buffer_size"]
    target_network_update_frequency = _param["target_network_update_freq"]
    final_epsilon = _param["final_epsilon"]
    epsilon_timesteps = _param["epsilon_timesteps"]
    reward_standardize = _param["reward_standardize"]
    optimizer = _param["optimizer"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    mixer_dict = {
        "qmix": "qmix",
        "vdn": "vdn",
        "iql": None
    }

    config = {
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": back_up_config,
        },
    }

    config.update(run)

    JointQ_Config.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": 100000,#buffer_size * episode_limit,  # in timesteps
            "train_batch_size": 6,#train_batch_episode,  # in sequence
            "target_network_update_freq": 100000,#episode_limit * target_network_update_frequency,  # in timesteps
            "learning_starts": 18000,#episode_limit * train_batch_episode,
            "lr": 0.0002, #lr if restore is None else 1e-10,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            },
            "mixer": mixer_dict[algorithm]
        })

    JointQ_Config["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    JointQ_Config["optimizer"] = optimizer
    JointQ_Config["training_intensity"] = None

    JQTrainer = JointQTrainer.with_updates(
        name=algorithm.upper(),
        default_config=JointQ_Config
    )

    map_name = '' #exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])


    import sys,os,time
    def fa_path(path,num):
        if num==0:
            return path
        return fa_path(os.path.dirname(os.path.abspath(path)),num-1)
    run_path=fa_path(os.path.abspath(__file__),6)
    sys.path.append(run_path)

    from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
    from ray.tune import register_env
    register_env(stop['env_reg_name'],
                              lambda _: RLlibEnvWrapper(stop['exp_info']["env_args"]).with_agent_groups(
                                  stop['grouping'], obs_space=stop['obs_space'],
                                  act_space=stop['act_space']))

    from group_callbacks import TrainerCallbacks
    model_path = restore['load_path']
    trainer_config = config
    _ = trainer_config.pop("evaluation_interval")

    trainer_config['env_config'] = config['model']['custom_model_config']['env_args']
    trainer_config['callbacks'] = TrainerCallbacks
    trainer_config['seed'] = restore['seed']

    trainer = JQTrainer( config=trainer_config)
    if model_path != '':
        trainer.restore(model_path)

    # restore.model_path-> save_path,load_path, train_iter, log_path
    dense_log = []
    pst_time = time.time()
    for i in range(restore['iter_this_run']):
        result = trainer.train()
        cur_time = time.time()
        print('time:', cur_time - pst_time, 'timestep: ', result['timesteps_total'])
        pst_time = cur_time

        print(result['custom_metrics'])
        print('pol,epi rew:', result['policy_reward_mean'], result['episode_reward_mean'])
        policy_rew = result['episode_reward_mean'] / 10  # num_agents

        if policy_rew > restore['best_rew']:
            trainer.save(os.path.join(restore['save_path'], 'rew_' + str(round(policy_rew, 4))))
        if (i + 1) % restore['iter_this_run'] == 0:
            trainer.save(os.path.join(restore['save_path'], 'iter_' + str(restore['num_iter']+i + 1)))

        if 'profit_mean' in result['custom_metrics'].keys():
            profit = round(result['custom_metrics']['profit_mean'], 4)
            equality = round(result['custom_metrics']['equality_mean'], 4)
            capability = round(result['custom_metrics']['capability_mean'], 4)
            dense_log.append({
                'iter': i, 'epi_rew': result['episode_reward_mean'], 'a_rew': policy_rew,
                'profit': profit, 'equality': equality, 'capability': capability,
                'timesteps': result['timesteps_total']
            })

    np.save(os.path.join(restore['save_path'],
                         f"dense_log_{restore['num_iter']}_{restore['num_iter'] + restore['iter_this_run']}.npy"),
            dense_log)

    return
