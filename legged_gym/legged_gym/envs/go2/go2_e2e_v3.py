"""
V3: The Precise Tracker (精准追踪)
牺牲一点速度，换取绝对精确的路线跟踪。
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV3Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            tracking_position = 8.0
            tracking_yaw = 4.0
            reach_pos_target_tight = 15.0


class GO2E2EV3CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v3_the_precise_tracker'
        experiment_name = 'go2_e2e_v3'
