from marllib import marl
from copy import deepcopy
from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl.common import dict_update
import sys
import yaml
from tabulate import tabulate
from ray.tune import register_env

SYSPARAMs = deepcopy(sys.argv)



def make_env(
        environment_name: str,
        env_config_path:str,
        **env_params
) :
    # env_config_file_path = os.path.join(os.path.dirname(__file__),
    #                                         "/home/game/test/MARLlib/marllib/envs/base_env/config/{}.yaml".format(environment_name))
    #
    # with open(env_config_file_path, "r") as f:
    with open(env_config_path, "r") as f:
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
    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == env_config["env"]:
                check_current_used_env_flag = True

    # print(tabulate(env_reg_ls,
    #                headers=['Env_Name', 'Check_Status', "Error_Log", "Config_File_Location", "Env_File_Location"],
    #                tablefmt='grid'))

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                env_config["env"]))

    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]
    register_env(env_reg_name, lambda _: ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
    env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    return env, env_config