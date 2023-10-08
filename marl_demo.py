import yaml
import os
import foundation
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.rllib.utils.framework import  try_import_torch
torch,_=try_import_torch()

config_path = os.path.join('./experiments', "marl_config_50_50.yaml")
with open(config_path, "r") as f:
    run_config = yaml.safe_load(f)

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from foundation.utils.marl_env_wrapper import MarlEnvWrapper
from experiments.utils import make_env
ENV_REGISTRY["magym"] = MarlEnvWrapper
env = make_env(environment_name="magym",env_config_path=config_path)

ray.init(local_mode=True,num_gpus=1)
ippo = marl.algos.iql(hyperparam_source="test")
# coma,mappo # ippo,ia2c #iql

#model = marl.build_model(env, mappo, {"core_arch": "lstm", "encode_layer": "128-128"})
model = marl.build_model(env, ippo, {"core_arch": "conv_lstm",
                                     'custom_model_config':run_config['agent_policy']['model']['custom_model_config']})

# start learning
# Trainer, config, cfg_run, model_cfg=\
ippo.fit(env, model, stop={'timesteps_total': 300000},
        #   local_mode=True,
         num_gpus=1,num_cpus=4,
          num_workers=4 , num_cpus_per_worker=1,
        #   # memory_per_worker= 8000*10*1000,#450 * 1024 * 1024,
        # # object_store_memory_per_worker=128 * 1024 * 1024,
        num_envs_per_worker=1,
         share_policy='group', checkpoint_freq=1)


