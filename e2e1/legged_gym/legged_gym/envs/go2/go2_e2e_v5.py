"""
V5: Positive-Only — Only positive rewards, no penalties except termination.
Hypothesis: Removing penalties entirely lets the E2E agent find its own locomotion.
This is the "purest" E2E — zero inductive bias from locomotion priors.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV5Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = True   # clip to positive only
        position_target_sigma = 0.5
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        heading_target_sigma = 1.0
        rew_duration = 2.0

        class scales(GO2E2ECfg.rewards.scales):
            # All penalties zeroed out
            torques = 0.0
            dof_pos_limits = 0.0
            dof_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            dof_acc = 0.0
            action_rate = 0.0
            orientation = 0.0
            collision = 0.0
            action_smoothness = 0.0
            power = 0.0
            feet_acceleration = 0.0
            feet_height = 0.0
            stop_yaw_vel = 0.0
            stand_still_pos = 0.0

            # Only positive task rewards
            task = 15.0
            catch_bonus = 10.0
            reach_landing_pos = 10.0
            track_ball_landing = 5.0
            reach_pos_target_tight = 8.0
            tracking_position = 3.0
            tracking_yaw = 1.0

            feet_air_time = 1.0
            exploration = 1.0
            stalling_penalty = 0.0
            termination = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            base_height = 0.0

class GO2E2EV5CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        student_reinforcing = False

    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v5_positive_only'
        experiment_name = 'go2_e2e_v5'
