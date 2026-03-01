"""
V1: Strongest Anti-overshoot — Higher velo_lim penalty, wider deceleration zone.
Hypothesis: V1 walked forward but overshot heavily. Stronger velo_lim
and lower velo_dir should fix this.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV1Cfg(GO2E2ECfg):
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

            # Direction: slightly lower (less rushing)
            velo_dir = 8.0             # reduced from 10
            exploration = 2.0          # reduced from 3

            # Anti-overshoot: STRONG
            velo_lim = -5.0            # 1.7x base

            # Task rewards: higher to compensate lower velo_dir
            task = 10.0
            catch_bonus = 5.0
            reach_landing_pos = 18.0
            track_ball_landing = 12.0
            reach_pos_target_tight = 12.0
            tracking_position = 5.0
            tracking_yaw = 2.0

            feet_air_time = 3.0
            stalling_penalty = 1.5
            stop_yaw_vel = -0.2

            termination = -0.
            stand_still_pos = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

class GO2E2EV1CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        pass
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v1_anti_overshoot'
        experiment_name = 'go2_e2e_v1'
