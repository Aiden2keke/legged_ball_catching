"""
V3: V5-style Maximum Forward — Based on the V5 config that also walked forward,
but with the new anti-overshoot fixes.
Hypothesis: V5 had stronger velo_dir (15) and walked forward with less overshoot.
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
            dof_pos = -2.0
            upright_bonus = 10.0

            # V5-style: strongest forward
            velo_dir = 15.0            # same as old V5
            exploration = 5.0          # same as old V5
            velo_lim = -5.0            # strong anti-overshoot

            task = 5.0
            catch_bonus = 2.0
            reach_landing_pos = 10.0
            track_ball_landing = 8.0
            reach_pos_target_tight = 8.0
            tracking_position = 2.0
            tracking_yaw = 1.0

            feet_air_time = 5.0        # strong gait
            stalling_penalty = 2.0
            stop_yaw_vel = -0.3

            termination = -0.
            stand_still_pos = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

class GO2E2EV3CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        pass
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v3_max_forward_decel'
        experiment_name = 'go2_e2e_v3'
