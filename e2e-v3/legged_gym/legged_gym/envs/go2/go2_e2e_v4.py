"""
V4: High Exploration — Higher noise, entropy, larger network.
Hypothesis: E2E needs more exploration to discover ball-catching strategy.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV4Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma = 0.5
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.8  # wider catch radius — easier
        heading_target_sigma = 1.0
        rew_duration = 2.0

        class scales(GO2E2ECfg.rewards.scales):
            torques = -0.0001
            dof_pos_limits = -3.0
            dof_vel = -0.001
            lin_vel_z = -1.5
            ang_vel_xy = -0.03
            dof_acc = -2.0e-7
            action_rate = -0.008
            orientation = -3.0
            collision = -0.5
            action_smoothness = -0.001
            power = -1e-4
            feet_acceleration = -1e-9
            feet_height = -0.5

            task = 15.0
            catch_bonus = 8.0
            reach_landing_pos = 8.0
            track_ball_landing = 5.0
            reach_pos_target_tight = 6.0
            tracking_position = 2.0
            tracking_yaw = 0.5

            feet_air_time = 1.0
            exploration = 2.0         # 4x, encourage movement
            stalling_penalty = 1.0
            stop_yaw_vel = -0.05
            termination = -0.
            stand_still_pos = -0.2
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            base_height = 0.0

class GO2E2EV4CfgPPO(GO2E2ECfgPPO):
    class policy(GO2E2ECfgPPO.policy):
        init_noise_std = 0.8         # higher exploration
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(GO2E2ECfgPPO.algorithm):
        entropy_coef = 0.02          # 2x higher entropy bonus
        learning_rate = 5.e-4
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v4_high_exploration'
        experiment_name = 'go2_e2e_v4'
