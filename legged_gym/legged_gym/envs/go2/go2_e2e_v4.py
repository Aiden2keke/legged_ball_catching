"""
V4: Task Maximizer
Overwhelm the policy with task completion rewards.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV4Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            task = 30.0
            catch_bonus = 15.0
            velo_dir = 10.0


class GO2E2EV4CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v4_task_maximizer'
        experiment_name = 'go2_e2e_v4'
