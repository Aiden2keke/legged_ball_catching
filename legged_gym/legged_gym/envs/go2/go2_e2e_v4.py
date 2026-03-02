"""
V4: Maximum Task & Catch — Similar to previous V4, extremely focused
on completing the task (catching), but using the V1 safe gait penalties
to prevent the massive death rate.
Hypothesis: Safe gait + massive task rewards = fast, stable catching.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV4Cfg(GO2E2ECfg):
    class commands(GO2E2ECfg.commands):
        max_speed = 2.0                # slightly faster allowed

    class rewards(GO2E2ECfg.rewards):
        position_target_sigma_tight = 0.8  # wider catch radius

        class scales(GO2E2ECfg.rewards.scales):
            velo_dir = 10.0            # lower to let task dominate

            # Task: VERY HIGH
            task = 25.0
            catch_bonus = 15.0
            reach_landing_pos = 15.0
            track_ball_landing = 10.0


class GO2E2EV4CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        entropy_coef = 0.015       # slightly more exploration for high task

    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v4_max_task_safe_gait'
        experiment_name = 'go2_e2e_v4'
