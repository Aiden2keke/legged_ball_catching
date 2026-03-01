"""
V2: Dense Tracking — High continuous tracking rewards, low sparse catch bonus.
Hypothesis: Dense rewards give better gradient signal than sparse catch bonus.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV2Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma = 0.5
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        heading_target_sigma = 1.0
        rew_duration = 2.0

        class scales(GO2E2ECfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -5.0
            dof_vel = -0.0015
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -4.0e-7
            action_rate = -0.01
            orientation = -5.0
            collision = -1.
            action_smoothness = -0.002
            power = -2e-4
            feet_acceleration = -1e-9
            feet_height = -1.0

            task = 5.0                  # low
            catch_bonus = 1.0           # very low sparse
            reach_landing_pos = 15.0    # high dense
            track_ball_landing = 10.0   # high dense
            reach_pos_target_tight = 10.0
            tracking_position = 3.0     # high
            tracking_yaw = 1.0

            feet_air_time = 1.0
            exploration = 0.5
            stalling_penalty = 0.5
            stop_yaw_vel = -0.1
            termination = -0.
            stand_still_pos = -0.3
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            base_height = 0.0

class GO2E2EV2CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v2_dense_tracking'
        experiment_name = 'go2_e2e_v2'
