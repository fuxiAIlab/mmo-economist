import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--stop-iter', type=int, default=300)
parser.add_argument('--algo', type=str, default='mappo')
args = parser.parse_args()

# ppo: 300 iter * 8000 timestep
# mappo,coma 300 iter * 8000
# iql: 12000 timestep, 200 iter

# sample runnign code:
# for seed in range(5):
#     python run.py --algo mappo --seed seed --stop-iter 300
#     python run.py --algo coma --seed seed --stop-iter 300
#     python run.py --algo iql --seed seed --stop-iter 200
#     python run.py --algo ippo --seed seed --stop-iter 300
#     python run.py --algo ia2c --seed seed --stop-iter 300

stop_iter = args.stop_iter
iter_this_run = 50 


exp_path = f'./{args.algo}_seed_{args.seed}'
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
def get_load_path(start_iter=0):
    lst = os.listdir(exp_path)
    best_rew=-1
    for i in lst:
        if i.startswith('rew_'):
            best_rew = max(best_rew, float(i[4:]))

    load_ckpt=os.listdir(
        os.path.join(exp_path+'/iter_'+str(start_iter))
    )[0]
    ckpt_dir=os.path.join(exp_path,'iter_'+str(start_iter),load_ckpt)
    ckpt_lst=[k for k in os.listdir(ckpt_dir) ]
    for ckpt in ckpt_lst:
        if  ckpt.startswith('check') and not ckpt.endswith('data'):

            ckpt_path=os.path.join(ckpt_dir,ckpt)
            return ckpt_path,best_rew

for  num_iter in range( stop_iter//iter_this_run):
    if num_iter>0:
        load_path,best_rew=get_load_path(num_iter*iter_this_run)
        load_path = '--load-path ' + load_path
    else:
        load_path=''
        best_rew=-1
    cmd=(f'python train_marl.py '
              f'--num-iter {num_iter*iter_this_run} --iter-this-run {iter_this_run} '
              f'--stop-iter {stop_iter} --seed {args.seed} --algo {args.algo} '
              f' {load_path} --save-path {exp_path} --log-path {exp_path} '
            f'--best-rew {best_rew}'
              )
    print(cmd)
    os.system(cmd)
