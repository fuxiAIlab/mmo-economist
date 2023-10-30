import os

cfgs=['config_20_80','config_50_50','config_80_20']
adjs=['fixed','random-asy','random-syn','greedy-pro','greedy-equ']
p2_adjs=['planner','planner_equ','planner_pro']
res_path='/data/private/yuanxi/get_exp_res'
f=open('test_cmd.sh','w')
for exp in os.listdir(os.path.join(res_path,'baseline')):
    adj,seed=exp.split('_')[-3],exp.split('_')[-1]
    exp_path=os.path.join(res_path,'baseline',exp)
    for cfg in cfgs:
        ckpt_path=os.path.abspath(
            os.path.join(exp_path,cfg,'iter_299/checkpoint_300/checkpoint-300'))
        for test_seed in range(5):
            f.write(f'python test_env.py --cfg {cfg} --adj {adj}'
                f' --seed {seed} --savedir baseline --restore {ckpt_path} '
                    f'--test-seed {test_seed} \n')

for exp in os.listdir(os.path.join(res_path,'planner')):
    if exp.split('_')[1] =='1':continue
    adj, seed = exp.split('_')[-3], exp.split('_')[-1]
    exp_path = os.path.join(res_path, 'planner', exp)
    for cfg in cfgs:
        runs_path=[ k for k in os.listdir(os.path.join(exp_path,cfg)) if k.startswith('rew')][0]
        runs_path=os.path.join(exp_path,cfg,runs_path)
        ckpt_path=os.path.abspath(os.path.join(runs_path,os.listdir(runs_path)[0],
                                  os.listdir(runs_path)[0].replace('_','-')))
        for test_seed in range(5):
            f.write(f'python test_env.py --cfg {cfg} --adj {adj}'
                f' --seed {seed} --savedir planner --restore {ckpt_path} '
                    f'--test-seed {test_seed} \n')

for exp in os.listdir(os.path.join(res_path,'ablation','ablation_equ')):
    if exp.split('_')[1] == '1': continue
    adj, seed = exp.split('_')[-3], exp.split('_')[-1]
    exp_path = os.path.join(res_path, 'ablation','ablation_equ', exp)
    for cfg in cfgs:
        runs_path=[ k for k in os.listdir(os.path.join(exp_path,cfg)) if k.startswith('rew')][0]
        runs_path=os.path.join(exp_path,cfg,runs_path)
        ckpt_path=os.path.abspath(os.path.join(runs_path,os.listdir(runs_path)[0],
                                  os.listdir(runs_path)[0].replace('_','-')))
        for test_seed in range(5):
            f.write(f'python test_env.py --cfg {cfg} --adj {adj}'
                    f' --seed {seed} --savedir ablation_equ --restore {ckpt_path} '
                    f'--test-seed {test_seed} \n')

for exp in os.listdir(os.path.join(res_path,'ablation','ablation_pro')):
    if exp.split('_')[1] == '1': continue
    adj, seed = exp.split('_')[-3], exp.split('_')[-1]
    exp_path = os.path.join(res_path, 'ablation','ablation_pro', exp)
    for cfg in cfgs:
        runs_path=[ k for k in os.listdir(os.path.join(exp_path,cfg)) if k.startswith('rew')][0]
        runs_path=os.path.join(exp_path,cfg,runs_path)
        ckpt_path=os.path.abspath(os.path.join(runs_path,os.listdir(runs_path)[0],
                                  os.listdir(runs_path)[0].replace('_','-')))
        for test_seed in range(5):
            f.write(f'python test_env.py --cfg {cfg} --adj {adj}'
                f' --seed {seed} --savedir ablation_pro  --restore {ckpt_path}'
                    f' --test-seed {test_seed} \n')

f.close()

