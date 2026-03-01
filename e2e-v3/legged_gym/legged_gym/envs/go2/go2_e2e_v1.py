"""
V1: Task-Heavy — Stronger task/catch rewards, weaker locomotion penalties.
Hypothesis: Robot needs more incentive to reach the ball, looser walking quality is OK.
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
            torques = -0.0001
            dof_pos_limits = -3.0
            dof_vel = -0.001
            lin_vel_z = -1.0
            ang_vel_xy = -0.02
            dof_acc = -2.0e-7
            action_rate = -0.005
            orientation = -2.0          # very loose
            collision = -0.5
            action_smoothness = -0.001
            power = -1e-4
            feet_acceleration = -1e-9
            feet_height = -0.5

            task = 20.0                 # 2x current
            catch_bonus = 10.0          # 2x current
            reach_landing_pos = 10.0    # 2x current
            track_ball_landing = 5.0    # 2.5x current
            reach_pos_target_tight = 8.0
            tracking_position = 2.0
            tracking_yaw = 1.0

            feet_air_time = 0.5
            exploration = 0.5
            stalling_penalty = 0.5
            stop_yaw_vel = -0.05
            termination = -0.
            stand_still_pos = -0.2
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            base_height = 0.0

class GO2E2EV1CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v1_task_heavy'
        experiment_name = 'go2_e2e_v1'
