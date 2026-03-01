"""
V3: Locomotion Quality — Stronger locomotion penalties for clean gait + moderate task.
Hypothesis: Better walking = faster convergence on reaching targets.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV3Cfg(GO2E2ECfg):
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
            torques = -0.0005          # 2.5x stricter
            dof_pos_limits = -10.0     # strict
            dof_vel = -0.003           # 2x stricter
            lin_vel_z = -3.0
            ang_vel_xy = -0.1          # 2x stricter
            dof_acc = -5.0e-7
            action_rate = -0.02        # 2x stricter
            orientation = -10.0        # 2x stricter
            collision = -2.0
            action_smoothness = -0.005 # 2.5x stricter
            power = -5e-4
            feet_acceleration = -1e-9
            feet_height = -2.0         # 2x stricter

            task = 10.0
            catch_bonus = 5.0
            reach_landing_pos = 5.0
            track_ball_landing = 2.0
            reach_pos_target_tight = 5.0
            tracking_position = 1.0
            tracking_yaw = 0.5

            feet_air_time = 2.0        # 2x, encourage proper gait
            exploration = 0.5
            stalling_penalty = 1.0     # 2x
            stop_yaw_vel = -0.2
            termination = -0.
            stand_still_pos = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            base_height = 0.0

class GO2E2EV3CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        student_reinforcing = False

    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v3_locomotion_quality'
        experiment_name = 'go2_e2e_v3'
