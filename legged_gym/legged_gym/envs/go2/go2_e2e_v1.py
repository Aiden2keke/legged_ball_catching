"""
V1: Faster Forward — Higher velo_dir and exploration.
Hypothesis: With V1's stable gait proven, we can push speed much harder
to solve the 'slow to reach far points' issue.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV1Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Speed constraints: VERY FAST
            velo_dir = 18.0            # 1.5x base speed incentive
            exploration = 5.0          # 1.6x base directional incentive
            velo_lim = -3.0            # slightly stronger anti-overshoot

            # Task tracking:
            reach_landing_pos = 18.0
            track_ball_landing = 12.0
            reach_pos_target_tight = 15.0


class GO2E2EV1CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v1_faster_forward'
        experiment_name = 'go2_e2e_v1'
