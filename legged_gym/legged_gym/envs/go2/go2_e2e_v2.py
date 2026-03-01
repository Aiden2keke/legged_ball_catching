"""
V2: Maximum Speed — Highest velo_dir and task rewards, but keeps
strict sim2sim constraints.
Hypothesis: Stronger forward speed incentive will overcome the strict
sim2sim gait penalties and force fast, clean running.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV2Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Strict sim2sim retained from base

            # Forward speed: VERY HIGH
            velo_dir = 18.0            # 1.5x base
            exploration = 5.0
            velo_lim = -5.0            # stronger anti-overshoot required for high speed

            # Task & Gait
            task = 15.0
            catch_bonus = 8.0
            reach_landing_pos = 20.0
            track_ball_landing = 15.0
            reach_pos_target_tight = 15.0

class GO2E2EV2CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v2_max_speed'
        experiment_name = 'go2_e2e_v2'
