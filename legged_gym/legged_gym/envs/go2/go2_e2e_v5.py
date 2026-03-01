"""
V5: High Catch Focus + Anti-overshoot — Strongest catch/task rewards
combined with strong anti-overshoot. Higher entropy for discovering
the complete catching strategy.
Hypothesis: With forward walk and anti-overshoot solved, the robot
needs maximum incentive to actually catch the ball.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV5Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma = 0.5
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.8  # wider catch radius
        heading_target_sigma = 1.0
        rew_duration = 2.0

        class scales(GO2E2ECfg.rewards.scales):
            torques = -0.0005
            dof_pos_limits = -10.0
            dof_vel = -0.003
            lin_vel_z = -3.0
            ang_vel_xy = -0.1
            dof_acc = -5.0e-7
            action_rate = -0.02
            orientation = -30.0
            collision = -2.0
            action_smoothness = -0.002
            power = -5e-4
            feet_acceleration = -1e-9
            feet_height = -2.0
            base_height = -20.0
            dof_pos = -3.0
            upright_bonus = 10.0

            velo_dir = 10.0
            exploration = 3.0
            velo_lim = -4.0

            # Task: HIGHEST
            task = 20.0
            catch_bonus = 10.0         # very high
            reach_landing_pos = 12.0
            track_ball_landing = 10.0
            reach_pos_target_tight = 15.0
            tracking_position = 3.0
            tracking_yaw = 1.0

            feet_air_time = 3.0
            stalling_penalty = 2.0
            stop_yaw_vel = -0.2

            termination = -0.
            stand_still_pos = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

class GO2E2EV5CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        entropy_coef = 0.015          # higher exploration
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v5_high_catch'
        experiment_name = 'go2_e2e_v5'
