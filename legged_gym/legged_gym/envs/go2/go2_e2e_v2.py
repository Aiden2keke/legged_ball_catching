"""
V2: Over-Tracker
Massive dense rewards for tracking, ensuring it never walks inaccurately.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV2Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Dense Tracking: VERY HIGH
            reach_landing_pos = 25.0
            track_ball_landing = 15.0
            tracking_position = 10.0
            tracking_yaw = 5.0


class GO2E2EV2CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v2_over_tracker'
        experiment_name = 'go2_e2e_v2'
