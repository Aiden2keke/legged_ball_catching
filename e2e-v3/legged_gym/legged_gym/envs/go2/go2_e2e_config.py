"""
End-to-End (E2E) Baseline Config for Go2 Ball Catching.

Key differences from GO2RoughCfg:
- num_observations = 48 (ball pos/vel instead of timer_left + commands)
- num_privileged_obs = 48 (same as obs, no teacher-student advantage)  
- Reward scales adjusted: no timer-gated rewards, continuous ball tracking
- Ball observation noise added
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2E2ECfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48  # 3 ang_vel + 3 gravity + 3 ball_pos + 3 ball_vel + 12 dof_pos + 12 dof_vel + 12 actions
        num_privileged_obs = 48  # Same as obs (no teacher-student for E2E baseline)
        num_privileged_latent = 32
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
        num_actors = 2
        obs_history_length = 5  # Must match ActorCritic default for ProprioceptiveEncoder input dim

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
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
            # E2E specific: ball observation noise
            ball_pos = 0.05   # meters
            ball_vel = 0.3    # m/s

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma = 0.5
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        heading_target_sigma = 1.0
        rew_duration = 2.0

        class scales(LeggedRobotCfg.rewards.scales):
            # === Locomotion penalties (reduced from original to prevent domination) ===
            torques = -0.0002
            dof_pos_limits = -5.0
            dof_vel = -0.0005 * 3
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.0e-7 * 2
            action_rate = -0.01
            orientation = -5.0        # was -40, too dominant
            collision = -1.
            action_smoothness = -0.002
            power = -2e-4
            feet_acceleration = -1e-9
            feet_height = -1.0

            # === E2E task rewards (moderate scales for stable training) ===
            track_ball_landing = 2.0   # was 5
            reach_landing_pos = 5.0    # was 20
            catch_bonus = 5.0          # was 50

            # Original position tracking (without timer gating)
            tracking_position = 1.0
            tracking_yaw = 0.5
            task = 10.0                # was 50
            reach_pos_target_tight = 5.0  # was 15

            # Locomotion quality
            feet_air_time = 1.0
            exploration = 0.5
            stalling_penalty = 0.5
            stop_yaw_vel = -0.1

            # Disabled / reduced
            termination = -0.
            stand_still_pos = -0.5     # was -5, too large without timer gate

            # Disabled
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            base_height = 0.0


class GO2E2ECfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5     # was 1.0, too high causing wild actions
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 5.e-4    # was 1e-3, reduce for stability
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        run_name = 'e2e_baseline'
        max_iterations = 7500
        experiment_name = 'go2_e2e'
        num_steps_per_env = 24
        save_interval = 50
