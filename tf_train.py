import yaml
import os
import foundation
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray import tune

from experiments import tf_models
from foundation.utils import plotting
from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.algorithms.ppo import PPO
from datetime import datetime
import tempfile
from ray.tune.logger import UnifiedLogger
ray.init(webui_host='127.0.0.1')
def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj', type=str, default='')# adjust type
    parser.add_argument('--restore', type=str, default='')# restore ckpt path
    parser.add_argument('--cfg', type=str, default='')# config_50_50.yaml
    parser.add_argument('--num-iter', type=int, default=-1)# train iter
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--phase', type=int, default=1)# 1/2
    res =parser.parse_args()
    return res

def on_episode_start(info):
    info["episode"].user_data["res"] = []
    # info["episode"].hist_data["res"] = []


def on_episode_step(info):
    # print(info['episode'].total_reward)

    if info['episode'].last_info_for('p').get('res') is not None:
        # if info['episode'].last_info_for('p').get('res')[0]>0:
        info["episode"].user_data["res"].append(info['episode'].last_info_for('p')['res'])



def on_episode_end(info):
    episode = info["episode"]
    res=np.array(episode.user_data["res"])
    idx=np.where(res[:,0]>0)[0] #idx of planner action
    equality=np.split(res[:, 1], idx + 1)
    equality=np.mean([np.mean(_) for _ in equality if len(_)>0])
    # print(f'profit: {res[idx,0]}, equality: {equality}, capability: {res[idx,2]}')

    info["episode"].custom_metrics["profit"] = res[idx[-1],0]#np.mean(res[idx,0])
    info["episode"].custom_metrics["equality"] = equality
    info["episode"].custom_metrics["capability"] = res[idx[-1],2]#np.mean(res[idx,2])
    info["episode"].custom_metrics["equXcap"] = np.mean(res[:,1]*res[:,2])
    # info["episode"].hist_data["res"] = np.mean(episode.user_data["res"])

def init_trainer(args):


    #config_path = os.path.join('./experiments', "config_50_50.yaml")
    # cfg_path='config_20_80.yaml'
    cfg_path=args.cfg
    config_path = os.path.join('./experiments', cfg_path+'.yaml')

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    trainer_config = run_configuration.get("trainer")


    if args.adj!='':
        run_configuration["env"]["adjustemt_type"]=args.adj
    if args.phase==2:
        run_configuration["general"]["train_planner"]=True
    else:
        run_configuration["general"]["train_planner"] = False
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
            "seed": args.seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": policy_mapping_fun,
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
            * trainer_config.get("num_envs_per_worker"),
        }
    )

    trainer_config['callbacks']={
        "on_episode_start": on_episode_start,
        "on_episode_step": on_episode_step,
        "on_episode_end": on_episode_end,
        # "on_sample_end": on_sample_end,
        # "on_postprocess_traj": on_postprocess_traj
        # "on_train_result": on_train_result,
                                 }

    exp_dir='runs/'
    log_dir=f"/phase_{args.phase}_{run_configuration['env']['adjustemt_type']}_{cfg_path[:12]}"
    save_dir=f"runs/phase_{args.phase}_{run_configuration['env']['adjustemt_type']}_seed_{args.seed}/{cfg_path[:12]}"

    trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config,
                         logger_creator=custom_log_creator(save_dir,'test'))
                         # logger_creator=custom_log_creator(exp_dir,'test'))
    return trainer,save_dir

def train(trainer,args,save_dir):
    NUM_ITERS = 400 if args.num_iter==-1 else args.num_iter
    save_metric='a' if args.phase==1 else 'p'
    metric_log=[]
    cur_best=0
    import time
    pst_time=time.time()
    trainer.save(save_dir)
    for iteration in range(NUM_ITERS):
        print(f'********** Iter : {iteration} **********')
        result = trainer.train()
        # import ipdb;ipdb.set_trace()
        #r1 = trainer.workers.local_worker().sampler.get_data().policy_batches
        #print(r1['a']['rewards'].sum(), result['episode_reward_mean'])

        cur_time = time.time()

        if save_metric in result['policy_reward_mean'].keys():
            if result['policy_reward_mean'][save_metric]>cur_best:
                cur_best=result['policy_reward_mean'][save_metric]
                trainer.save(f"{save_dir}/rew_{round(cur_best,4)}")
            iter_time=round(cur_time-pst_time,4)
            episode_reward_mean=round(result.get('episode_reward_mean'),6)
            a_rew=round(result['policy_reward_mean']['a'],6)
            p_rew=round(result['policy_reward_mean']['p'],6)

            if 'profit_mean' in result['custom_metrics'].keys():
                profit=round(result['custom_metrics']['profit_mean'],6)
                equality=round(result['custom_metrics']['equality_mean'],6)
                capability=round(result['custom_metrics']['capability_mean'],6)
                equxcap=round(result['custom_metrics']['equXcap_mean'],6)
                print(f"time: {iter_time} epi_rew: {episode_reward_mean} a_rew:{a_rew} ",
                      f" p_rew:{p_rew}, epi_len: {result['episode_len_mean'] }",
                      f" pro:{profit} equ:{equality} cap:{capability} ,prod:{equxcap}")
                metric={'iter':iteration,'epi_len':result['episode_len_mean'],
                        'epi_rew':episode_reward_mean,
                        'a_rew':a_rew,'p_rew':p_rew,
                        'profit':profit,'equlity':equality,
                        'capability':capability,'equXcap':equxcap
                        }
                metric_log.append(metric)
            pst_time=cur_time
        else:
            print(f"episode_reward_mean: {result.get('episode_reward_mean')}")
        if iteration  % 10 == 9:
            trainer.save(f"{save_dir}/iter_{iteration}")
            print(f"save ckpt at iter {iteration}")
    np.save(os.path.join(save_dir,'metric.npy'),metric_log)

if __name__=="__main__":
    args=parse_args()
    trainer,save_dir=init_trainer(args)
    if args.restore!='':
        trainer._restore(args.restore)
        # trainer._restore('dir_ckpt_random-asy/iter_399/checkpoint_400/checkpoint-400')
    train(trainer,args,save_dir)