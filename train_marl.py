import yaml
import os
import foundation
import numpy as np
import matplotlib.pyplot as plt
import ray


from ray.rllib.utils.framework import  try_import_torch
torch,_=try_import_torch()
# print(torch.cuda.is_available())
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True

config_path = os.path.join('./experiments', "marl.yaml")
with open(config_path, "r") as f:
    run_configuration = yaml.safe_load(f)

from marllib import marl
from copy import deepcopy
from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl.common import dict_update
import sys
from tabulate import tabulate
from ray.tune import register_env
SYSPARAMs = deepcopy(sys.argv)

def make_env(
        environment_name: str,
        **env_params
) :
    env_config_file_path='./experiments/marl.yaml'
    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # update function-fixed config
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], env_params, True)

    # user commandline config
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], user_env_args, True)
    env_config_dict["env_args"]["map_name"] = 'dd'#map_name
    env_config_dict["force_coop"] = False #force_coop

    # combine with exp running config
    env_config = marl.set_ray(env_config_dict)

    # initialize env

    env_reg_name = env_config["env"] + "_" #+ env_config["env_args"]["map_name"]

    register_env(env_reg_name, lambda _: ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
    env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    return env, env_config

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
ENV_REGISTRY["magym"] = RLlibEnvWrapper
env = make_env(environment_name="magym")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save-path", type=str, default="")
parser.add_argument("--load-path", type=str, default="")
parser.add_argument("--log-path", type=str, default="")

parser.add_argument("--algo", type=str, default="mappo")# coma, iql, mappo
parser.add_argument("--seed", type=int, default=10)

parser.add_argument("--num-iter", type=int, default=10)#start from iter xx
parser.add_argument("--stop-iter", type=int, default=10)
parser.add_argument("--iter-this-run", type=int, default=10)
parser.add_argument("--best-rew", type=float, default=0.0)

args = parser.parse_args()

# env = marl.make_env(environment_name="magym", map_name="Checkers")
# pick mappo algorithms
if args.algo == "coma":
    algo = marl.algos.coma(hyperparam_source="test")
elif args.algo == "iql":
    algo = marl.algos.iql(hyperparam_source="test")
elif args.algo == "mappo":
    algo = marl.algos.mappo(hyperparam_source="test")

#model = marl.build_model(env, mappo, {"core_arch": "lstm", "encode_layer": "128-128"})
model = marl.build_model(env, algo, {"core_arch": "mlp",
                                     'custom_model_config':run_configuration['agent_policy']['model']['custom_model_config']})
# start learning
restore_cfg={
    'load_path': args.load_path,
    'save_path': args.save_path,
    'log_path': args.log_path,
    'num_iter': args.num_iter, # start from iter xx
    'stop_iter': args.stop_iter, # 300 total
    'iter_this_run': args.iter_this_run,
    'best_rew': args.best_rew,
    'seed': args.seed
}
algo.fit(env, model, stop={'episode_reward_mean': 400, 'timesteps_total': 3000000},
          #local_mode=True,
         num_gpus=1,num_cpus=4,
          num_workers=4, num_cpus_per_worker=1,
        num_envs_per_worker=1,
         share_policy='group',# checkpoint_freq=1,
         restore_path=restore_cfg
         )

