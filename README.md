<!--
 * @Author: Shiwei Zhao
 * @Date: 2023-08-18 10:10:33
 * @FilePath: \mmo-economist\README.md
 * Copyright (c) 2023 by NetEase, Inc., All Rights Reserved.
-->
# mmo-economist
MMO经济学家（数值坑）


## Requirements
The code has been tested running under Python 3.8.16, with the following packages installed (along with their dependencies):
- gym==0.21
- tensorflow==1.14
- ray[rllib]==0.8.4
- importlib-metadata==4.13.0

## Components
- player
  - market 游戏中的自由交易市场，玩家与玩家之间使用Token交易，只流通tradable=True的Resource
  - shop 游戏中的官方商店，玩家与NPC之间使用Token交易，目前流通除Token外全部Resource
  - recharge 充值，玩家通过Currency兑换Token
  - upgrade 升级，玩家消耗相应资源来升级获取数值Capability
  - task 任务，通过移动-采集来模拟玩家打本获取资源的过程
- planner
  - readjustment 间隔adjustment_period个timestep周期性生成新的投放计划  
## Running
- demo.ipynb 环境示例
- demo_rllib_yaml.ipynb 复现the-ai-economist baseline
