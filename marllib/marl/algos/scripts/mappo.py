# MIT License
import os.path

import numpy as np
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
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.scripts.coma import restore_model
import json
from typing import Any, Dict
from ray.tune.analysis import ExperimentAnalysis


def run_mappo(model: Any, exp: Dict, run: Dict, env: Dict,
              stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the Multi-Agent Proximal Policy Optimisation (MAPPO) algorithm using Ray RLlib.
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
    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    ModelCatalog.register_custom_model(
        "Centralized_Critic_Model", model)

    _param = AlgVar(exp)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = 3200 #_param["batch_episode"] * env["episode_limit"]
    if "fixed_batch_timesteps" in exp:
        train_batch_size = exp["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    batch_mode = _param["batch_mode"]
    lr = _param["lr"]
    clip_param = _param["clip_param"]
    vf_clip_param = _param["vf_clip_param"]
    use_gae = _param["use_gae"]
    gae_lambda = _param["lambda"]
    kl_coeff = _param["kl_coeff"]
    num_sgd_iter = _param["num_sgd_iter"]
    vf_loss_coeff = _param["vf_loss_coeff"]
    entropy_coeff = _param["entropy_coeff"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    config = {
        'rollout_fragment_length': 600,
        "batch_mode": 'truncate_episodes',#batch_mode,
        "train_batch_size": 8000, #train_batch_size,
        "sgd_minibatch_size": 4000, #sgd_minibatch_size,
        "lr": 0.0003,#lr if restore is None else 1e-10,
        "entropy_coeff": 0.0025,#entropy_coeff,
        "num_sgd_iter": 3,#num_sgd_iter,
        "clip_param": 0.2,#clip_param,
        "use_gae": True, #use_gae,
        "lambda": 0.98,#gae_lambda,
        "vf_loss_coeff": 0.05, #vf_loss_coeff,
        "kl_coeff": 0.0, #kl_coeff,
        "vf_clip_param": 50.0, #vf_clip_param,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "custom_model_config": back_up_config,
        },
    }
    config.update(run)

    algorithm = exp["algorithm"]
    map_name = ''#exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])
    # model_path = restore_model(restore, exp)
    model_path=restore['load_path']
    import sys, os,time
    def fa_path(path, num):
        if num == 0:
            return path
        return fa_path(os.path.dirname(os.path.abspath(path)), num - 1)

    # sys.path.append(os.path.dirname(os.path.abspath('./')))
    run_path = fa_path(os.path.abspath(__file__), 6)
    sys.path.append(run_path)

    from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
    from callbacks import TrainerCallbacks
    trainer_config = config
    _=trainer_config.pop("env")
    _=trainer_config.pop("evaluation_interval")
    trainer_config['multiagent']['policies'] = {
        'shared_policy':(
            None,
            config['model']['custom_model_config']['space_obs'],
            config['model']['custom_model_config']['space_act']['a'],
            config['model']
        )
    }
    trainer_config['env_config']=config['model']['custom_model_config']['env_args']
    trainer_config['callbacks']=TrainerCallbacks
    trainer_config['seed']=restore['seed']

    trainer=MAPPOTrainer(env=RLlibEnvWrapper,config=trainer_config)
    if model_path != '':
        trainer.restore(model_path)

    # restore.model_path-> save_path,load_path, train_iter, log_path
    dense_log=[]
    pst_time=time.time()
    for i in range(restore['iter_this_run']):
        result=trainer.train()
        cur_time=time.time()
        print('time:',cur_time-pst_time,'timestep: ',result['timesteps_total'])
        pst_time=cur_time

        print(result['custom_metrics'])
        print('pol,epi rew:',result['policy_reward_mean'],result['episode_reward_mean'])
        policy_rew=result['episode_reward_mean']/10 #num_agents

        if policy_rew > restore['best_rew']:
            trainer.save(os.path.join(restore['save_path'],'rew_'+str(round(policy_rew,4))))
        if (i+1)%restore['iter_this_run']==0:
            trainer.save(os.path.join(restore['save_path'],'iter_'+str(restore['num_iter']+i+1)))

        if 'profit_mean' in result['custom_metrics'].keys():
            profit = round(result['custom_metrics']['profit_mean'], 4)
            equality = round(result['custom_metrics']['equality_mean'], 4)
            capability = round(result['custom_metrics']['capability_mean'], 4)
            dense_log.append({
                'iter':i,'epi_rew':result['episode_reward_mean'],'a_rew':policy_rew,
                'profit':profit,'equality':equality,'capability':capability,
                 'timesteps':result['timesteps_total']
            })

    np.save(os.path.join(restore['save_path'],
                         f"dense_log_{restore['num_iter']}_{restore['num_iter']+restore['iter_this_run']}.npy"),
            dense_log)
    return
