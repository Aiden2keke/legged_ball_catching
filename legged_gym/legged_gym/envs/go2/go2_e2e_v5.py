"""
V5: Dense Tracking Dominant — High dense tracking rewards throughout
the approach, guiding the robot strictly to the landing zone.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV5Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Strict sim2sim retained from base

            # Speed
            velo_dir = 10.0
            exploration = 3.0
            velo_lim = -3.0

            # Task: Dense tracking VERY HIGH
            task = 10.0
            catch_bonus = 5.0
            reach_landing_pos = 30.0   # extremely high dense distance reward
            track_ball_landing = 20.0  # extremely high dense tracking
            reach_pos_target_tight = 12.0
            tracking_position = 8.0
            tracking_yaw = 3.0

            # Gait
            feet_air_time = 5.0
            stalling_penalty = 2.0

class GO2E2EV5CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v5_dense_tracking'
        experiment_name = 'go2_e2e_v5'
