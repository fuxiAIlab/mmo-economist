import os
import numpy as np

cfgs=['config_20_80','config_50_50','config_80_20']
adjs=['fixed','random-asy','random-syn','greedy-pro','greedy-equ']
tot_adjs=['fixed','random-asy','random-syn','greedy-pro','greedy-equ','planner','ablation_equ','ablation_pro']
p2_adjs=['planner','planner_equ','planner_pro']
metrics=['profit','equality','capability','last_graduate','mean_graduate','equXcap',
         'period_equality','last_equality']
# metrics=['profit','equality','capability','last_graduate','mean_graduate','equXcap','new_equality']
seeds=['seed_0','seed_1','seed_2','seed_3','seed_4']
test_seeds=['_'+str(i) for i in range(0,5)]
baseline='baseline'
summary=[]

merge_res={
    cfg:{
        adj:{
            metric:[] for metric in metrics
        } for adj in tot_adjs
    } for cfg in cfgs
}

for cfg in cfgs:
    grad_req=np.arange(10)*int(cfg[7])+int(cfg[7])

    for adj in adjs:
        for seed in seeds:
            for test_seed in test_seeds:
                '''
                res=np.load(os.path.join(baseline,cfg,adj,f'dense_logs_{seed}{test_seed}.npy'),allow_pickle=True).item()
                recharge = np.sum([len(res[0]['Recharge'][j]) for j in range(len(res[0]['Upgrade']))])
                upgrade = np.sum([len(res[0]['Upgrade'][j]) for j in range(len(res[0]['Upgrade']))])
                print(
                    f'{cfg}\t{adj} {seed} {test_seed}\t: Recharge: {recharge} Upgrade: {upgrade} episode_length: {len(res[0]["Upgrade"])}')
                continue
                '''
                res=np.load(os.path.join(baseline,cfg,adj,f'dense_logs_{seed}{test_seed}.npy'),allow_pickle=True).item()
                # if len(res[0]['Upgrade'])>2000:continue
                reset_idx=np.where(np.array(res[0]['period_step'])<1)[0][1:].tolist()+[len(res[0]['equality'])]
                reset_idx=np.array(reset_idx)-1

                grad_cnt=0
                mean_graduate=[]

                each_upgrade={i:0 for i in range(10) }
                for i in range(len(res[0]['Upgrade'])):
                    for j in res[0]['Upgrade'][i]:
                        each_upgrade[j['upgrader']]+=1
                    if i == reset_idx[grad_cnt]:
                        mean_graduate.append(
                            np.mean([int(each_upgrade[k] > grad_req[grad_cnt])
                                     for k in each_upgrade.keys()])
                        )
                        grad_cnt+=1
                upgrade=[ len(res[0]['Upgrade'][i]) for i in range(len(res[0]['Upgrade']))]
                tot_upgrade=np.cumsum(upgrade)
                charge=[ len(res[0]['Recharge'][i]) for i in range(len(res[0]['Recharge']))]
                equality=res[0]['equality']

                merge_res[cfg][adj]['profit'].append(res[0]['profitability'][-1])
                merge_res[cfg][adj]['equality'].append(np.mean(res[0]['equality']))
                merge_res[cfg][adj]['capability'].append(
                    np.mean([each_upgrade[k] for k in each_upgrade.keys() ])
                )
                merge_res[cfg][adj]['last_graduate'].append(
                    np.mean([int(each_upgrade[k]>grad_req[-1]) for k in each_upgrade.keys()])
                )
                merge_res[cfg][adj]['mean_graduate'].append(np.mean(mean_graduate))
                merge_res[cfg][adj]['equXcap'].append(
                    np.mean(res[0]['equality']*np.sqrt(tot_upgrade/10)/np.sqrt(int(cfg[7:9]))))
                # merge_res[cfg][adj]['new_equality'].append(np.mean(tot_upgrade*np.array(equality)))
                merge_res[cfg][adj]['period_equality'].append(
                    np.mean(tot_upgrade[reset_idx] * np.array(equality)[reset_idx]))
                merge_res[cfg][adj]['last_equality'].append(
                    np.mean(tot_upgrade[reset_idx[-1]] * np.array(equality)[reset_idx[-1]]))

for cfg in cfgs:
    for adj in adjs:
        for metric in metrics:
            mean,std=np.mean(merge_res[cfg][adj][metric]),np.std(merge_res[cfg][adj][metric])
            summary.append([cfg,adj,metric,mean,std])

for cfg in cfgs:
    grad_req=np.arange(10)*int(cfg[7])+int(cfg[7])

    for adj in ['planner','ablation_equ','ablation_pro']:
        for seed in seeds:
            for test_seed in test_seeds:
                '''
                res=np.load(os.path.join(adj,cfg,'planner',f'dense_logs_{seed}{test_seed}.npy'),allow_pickle=True).item()
                recharge = np.sum([len(res[0]['Recharge'][j]) for j in range(len(res[0]['Upgrade']))])
                upgrade = np.sum([len(res[0]['Upgrade'][j]) for j in range(len(res[0]['Upgrade']))])
                
                print(f'{cfg}\t{adj} {seed} {test_seed}\t: Recharge: {recharge} Upgrade: {upgrade} episode_length: {len(res[0]["Upgrade"])}')
                continue
                '''
                res=np.load(os.path.join(adj,cfg,'planner',f'dense_logs_{seed}{test_seed}.npy'),allow_pickle=True).item()
                # if len(res[0]['Upgrade']) > 2000: continue
                reset_idx=np.where(np.array(res[0]['period_step'])<1)[0][1:].tolist()+[len(res[0]['equality'])]
                reset_idx=np.array(reset_idx)-1

                grad_cnt=0
                mean_graduate=[]
                each_upgrade={i:0 for i in range(10) }
                for i in range(len(res[0]['Upgrade'])):
                    for j in res[0]['Upgrade'][i]:
                        each_upgrade[j['upgrader']]+=1
                    if i == reset_idx[grad_cnt]:
                        mean_graduate.append(
                            np.mean([int(each_upgrade[k] > grad_req[grad_cnt])
                                     for k in each_upgrade.keys()])
                        )
                        grad_cnt+=1
                upgrade=[ len(res[0]['Upgrade'][i]) for i in range(len(res[0]['Upgrade']))]
                tot_upgrade=np.cumsum(upgrade)
                charge=[ len(res[0]['Recharge'][i]) for i in range(len(res[0]['Recharge']))]
                equality=res[0]['equality']

                merge_res[cfg][adj]['profit'].append(res[0]['profitability'][-1])
                merge_res[cfg][adj]['equality'].append(np.mean(res[0]['equality']))
                merge_res[cfg][adj]['capability'].append(
                    np.mean([each_upgrade[k] for k in each_upgrade.keys() ])
                )
                merge_res[cfg][adj]['last_graduate'].append(
                    np.mean([int(each_upgrade[k]>grad_req[-1]) for k in each_upgrade.keys()])
                )
                merge_res[cfg][adj]['mean_graduate'].append(np.mean(mean_graduate))
                merge_res[cfg][adj]['equXcap'].append(
                    np.mean(res[0]['equality']*np.sqrt(tot_upgrade/10)/np.sqrt(int(cfg[7:9]))))
                # merge_res[cfg][adj]['new_equality'].append(np.mean(tot_upgrade * np.array(equality)))
                merge_res[cfg][adj]['period_equality'].append(
                    np.mean(tot_upgrade[reset_idx] * np.array(equality)[reset_idx]))
                merge_res[cfg][adj]['last_equality'].append(
                    np.mean(tot_upgrade[reset_idx[-1]] * np.array(equality)[reset_idx[-1]]))

for cfg in cfgs:
    for adj in ['planner','ablation_equ','ablation_pro']:
        for metric in metrics:
            mean,std=np.mean(merge_res[cfg][adj][metric]),np.std(merge_res[cfg][adj][metric])
            summary.append([cfg,adj,metric,mean,std])



# import ipdb;ipdb.set_trace()


f=open('summary.csv','w')
for res in summary:f.write(f'{res[0]},{res[1]},{res[2]},{round(res[3],4)},{round(res[4],4)}\n')
f.close()
# for res in summary:
#     print(f'{res[0]}\t{res[1]}\t{res[2]}\t'
#             f'{round(res[3], 4)}\t{round(res[4], 4)}\n')
#     f.write(f'{res[0]}\t{res[1]}\t{res[2]}\t{round(res[3],4)}\t{round(res[4],4)}\n')


