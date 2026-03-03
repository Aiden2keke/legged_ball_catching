"""
V4: The Task Master (任务驱动)
给出极其夸张的抓球奖励，看看强化学习能否自己悟出诀窍。
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV4Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            catch_bonus = 30.0
            task = 30.0


class GO2E2EV4CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v4_the_task_master'
        experiment_name = 'go2_e2e_v4'
