"""
V4: Balanced High Task & Gait — High priority on catching and
reaching landing pos, but maintains strong gait constraints.
Hypothesis: With strict gait enforced, we can afford higher task rewards
to force the policy to find a valid running solution to catch the ball.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV4Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Strict sim2sim retained from base

            # Speed
            velo_dir = 12.0
            exploration = 3.0
            velo_lim = -3.0

            # Task: VERY HIGH
            task = 20.0
            catch_bonus = 10.0
            reach_landing_pos = 20.0
            track_ball_landing = 15.0
            reach_pos_target_tight = 15.0
            tracking_position = 5.0
            tracking_yaw = 2.0

            # Gait
            feet_air_time = 5.0
            stalling_penalty = 2.0

class GO2E2EV4CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        entropy_coef = 0.015       # slightly more exploration for high task

    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v4_balanced_high_task'
        experiment_name = 'go2_e2e_v4'
