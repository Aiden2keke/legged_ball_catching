"""
End-to-End (E2E) Baseline Config for Go2 Ball Catching.

Based on V1 forward-walking success (velo_dir=10, exploration=3).
Added velo_lim to prevent overshooting, and smooth deceleration in _reward_velo_dir.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2E2ECfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48  # 3 ang_vel + 3 gravity + 3 ball_pos + 3 ball_vel + 12 dof_pos + 12 dof_vel + 12 actions
        num_privileged_obs = 48 + 3 + 63 + 12 + 12 + 12  # obs + lin_vel + adapt_obs + torques + dof_acc + contacts = 150 (TS)
        num_privileged_latent = 32
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
        num_actors = 2
        obs_history_length = 5

    class commands(LeggedRobotCfg.commands):
        max_speed = 2.0
        landing_time_min = 1.0
        landing_time_max = 10.0
        curriculum = True
        ball_dist_min = 1.5
        ball_dist_max = 5.0
        ball_apex_max_start = 5.0
        ball_apex_max_end = 3.5
        class ranges(LeggedRobotCfg.commands.ranges):
            pos_1 = [1.0, 4.0]
            pos_2 = [-3.14, 3.14]
            heading = [-0.5, 0.5]
            use_polar = True

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1, 'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1, 'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.,
            'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.,
            'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "Head_lower"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            ball_pos = 0.05
            ball_vel = 0.3

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32
        only_positive_rewards = False
        position_target_sigma = 0.5
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        heading_target_sigma = 1.0
        rew_duration = 2.0

        class scales(LeggedRobotCfg.rewards.scales):
            # === Locomotion penalties (V1 Stable Sim2Sim) ===
            torques = -0.0001
            dof_pos_limits = -3.0
            dof_vel = -0.002            # 翻倍，约束关节速度，防止乱飞
            lin_vel_z = -2.0            # 增强Z轴惩罚，抑制爬行趋势
            ang_vel_xy = -0.05
            dof_acc = -1.0e-6           # 增强加速度惩罚，使动作更加平滑
            action_rate = -0.01
            orientation = -5.0          # 增强姿态约束，强烈惩罚俯仰和横滚，保持正着走
            collision = -0.5
            action_smoothness = -0.005  # 增强动作平滑度约束
            power = -1e-4
            feet_acceleration = -1e-9
            feet_height = -0.5
            base_height = -2.0

            # === Direction & Anti-overshoot ===
            velo_dir = 10.0             # Encourage moving to target
            velo_lim = -12.0            # 从V2吸取经验：极其实严苛的限速，强行防过冲

            # === Task Tracking ===
            task = 20.0                 # 2x current
            catch_bonus = 12.0          # 2x current
            reach_landing_pos = 12.0    # 2x current
            track_ball_landing = 7.0    # 2.5x current
            reach_pos_target_tight = 8.0
            tracking_position = 2.0
            tracking_yaw = 1.0

            # === Gait quality ===
            feet_air_time = 3.0         # Fix crawling: must be high enough to encourage trots!
            exploration = 0.5
            stalling_penalty = 0.5
            stop_yaw_vel = -0.05
            termination = -0.
            stand_still_pos = -3.0      # 从V2吸取经验：强制在目标点冻结姿态，防过冲
            tracking_lin_vel = 1
            tracking_ang_vel = 0.5


class GO2E2ECfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        num_learning_epochs = 5

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        run_name = 'e2e_baseline'
        max_iterations = 10000
        experiment_name = 'go2_e2e'
        num_steps_per_env = 24
        save_interval = 50
