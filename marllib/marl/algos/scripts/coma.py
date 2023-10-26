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
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.core.CC.coma import COMATrainer
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.utils.log_dir_util import available_local_dir
import json
from typing import Any, Dict
from ray.tune.analysis import ExperimentAnalysis

import numpy as np
def restore_model(restore: Dict, exp: Dict):
    
    return restore["load_path"] # hack for now
    if restore is not None:
        with open(restore["params_path"], 'r') as JSON:
            raw_exp = json.load(JSON)
            raw_exp = raw_exp["model"]["custom_model_config"]['model_arch_args']
            check_exp = exp['model_arch_args']
            if check_exp != raw_exp:
                raise ValueError("is not using the params required by the checkpoint model")
        model_path = restore["model_path"]
    else:
        model_path = None

    return model_path


def run_coma(model: Any, exp: Dict, run: Dict, env: Dict,
             stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the Counterfactual Multi-Agent Policy Gradients (COMA) algorithm using Ray RLlib.
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
        "Centralized_Critic_Model", model)

    _param = AlgVar(exp)

    train_batch_size = _param["batch_episode"] * env["episode_limit"]
    if "fixed_batch_timesteps" in exp:
        train_batch_size = exp["fixed_batch_timesteps"]
    episode_limit = env["episode_limit"]

    batch_mode = _param["batch_mode"]
    lr = _param["lr"]
    use_gae = _param["use_gae"]
    gae_lambda = _param["lambda"]
    vf_loss_coeff = _param["vf_loss_coeff"]
    entropy_coeff = _param["entropy_coeff"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    config = {
        "train_batch_size": 8000, #train_batch_size,
        "batch_mode":'truncate_episodes',#batch_mode,
        "use_gae": True, #use_gae,
        "lambda": 0.98,#gae_lambda,
        "vf_loss_coeff": 0.05,#vf_loss_coeff,
        "entropy_coeff": 0.0025,#entropy_coeff,
        "lr": 0.0003,#lr if restore is None else 1e-10,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": back_up_config,
        },
    }

    config.update(run)

    algorithm = exp["algorithm"]
    map_name = ''#exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    # model_path = restore_model(restore, exp)

    model_path = restore['load_path']
    import sys, os, time
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
    _ = trainer_config.pop("env")
    _ = trainer_config.pop("evaluation_interval")
    trainer_config['multiagent']['policies'] = {
        'shared_policy': (
            None,
            config['model']['custom_model_config']['space_obs'],
            config['model']['custom_model_config']['space_act']['a'],
            config['model']
        )
    }
    trainer_config['env_config'] = config['model']['custom_model_config']['env_args']
    trainer_config['callbacks'] = TrainerCallbacks
    trainer_config['seed'] = restore['seed']

    trainer = COMATrainer(env=RLlibEnvWrapper, config=trainer_config)
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
            trainer.save(os.path.join(restore['save_path'], 'iter_' + str(restore['num_iter']+ i + 1)))

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

