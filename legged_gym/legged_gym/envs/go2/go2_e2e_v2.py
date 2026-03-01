"""
V2: Dense Tracking + Deceleration — High dense tracking rewards
to guide the robot precisely to the landing point, with anti-overshoot.
Hypothesis: Dense reward gradient helps the robot learn fine positioning.
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
            # Locomotion: slightly relaxed
            torques = -0.0003
            dof_pos_limits = -8.0
            dof_vel = -0.002
            lin_vel_z = -2.5
            ang_vel_xy = -0.08
            dof_acc = -4.0e-7
            action_rate = -0.015
            orientation = -25.0
            collision = -1.5
            action_smoothness = -0.001
            power = -3e-4
            feet_acceleration = -1e-9
            feet_height = -1.5
            base_height = -15.0
            dof_pos = -2.0
            upright_bonus = 8.0

            velo_dir = 10.0
            exploration = 3.0
            velo_lim = -3.0

            # Dense tracking: HIGH
            task = 6.0
            catch_bonus = 2.0
            reach_landing_pos = 25.0   # very high
            track_ball_landing = 18.0  # very high
            reach_pos_target_tight = 15.0
            tracking_position = 8.0    # high position tracking
            tracking_yaw = 2.0

            feet_air_time = 2.0
            stalling_penalty = 1.0
            stop_yaw_vel = -0.1

            termination = -0.
            stand_still_pos = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

class GO2E2EV2CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        pass
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v2_dense_decel'
        experiment_name = 'go2_e2e_v2'
