"""
End-to-End (E2E) Baseline Environment for Ball Catching.

This environment removes the Kalman Filter and time-conditioned reward masking
from the observation/reward pipeline. Instead, the policy receives raw (noisy)
ball position and velocity in the robot's body frame and must learn to implicitly
infer the ball trajectory and landing position.
"""

import torch
import numpy as np
from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi, yaw_quat


class LeggedRobotE2E(LeggedRobot):
    """End-to-End baseline: raw noisy ball state → joint actions, no physics priors."""

    def _init_buffers(self):
        super()._init_buffers()
        # ---- Ball trajectory simulation buffers ----
        # Ball launch position (world frame)
        self.ball_launch_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        # Ball launch velocity (world frame)
        self.ball_launch_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        # Current simulated ball position (world frame)
        self.ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        # Current simulated ball velocity (world frame)
        self.ball_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        # Time elapsed since ball was launched
        self.ball_time_elapsed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Total flight time for the ball
        self.ball_flight_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Noisy ball observations (body frame)
        self.ball_pos_body_noisy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel_body_noisy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

    def _resample_commands(self, env_ids):
        """Sample ball landing positions and compute ballistic launch trajectories.
        
        We keep the same landing position sampling as the original, but additionally
        compute a plausible launch trajectory for the ball so we can simulate its
        flight path and provide raw ball observations.
        """
        tbd_envs = env_ids.clone()
        while len(tbd_envs) > 0:
            _target_pos1 = torch_rand_float(self.command_ranges["pos_1"][0], self.command_ranges["pos_1"][1], (len(tbd_envs), 1), device=self.device)
            _target_pos2 = torch_rand_float(self.command_ranges["pos_2"][0], self.command_ranges["pos_2"][1], (len(tbd_envs), 1), device=self.device)
            _target_heading = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(tbd_envs), 1), device=self.device)
            if self.cfg.commands.ranges.use_polar:
                self.position_targets[tbd_envs, 0:1] = self.env_origins[tbd_envs, 0:1] + _target_pos1 * torch.cos(_target_pos2)
                self.position_targets[tbd_envs, 1:2] = self.env_origins[tbd_envs, 1:2] + _target_pos1 * torch.sin(_target_pos2)
            else:
                self.position_targets[tbd_envs, 0:1] = self.env_origins[tbd_envs, 0:1] + _target_pos1
                self.position_targets[tbd_envs, 1:2] = self.env_origins[tbd_envs, 1:2] + _target_pos2
            self.position_targets[tbd_envs, 2] = self.env_origins[tbd_envs, 2] + 0.4

            pos_diff = self.position_targets[tbd_envs] - self.root_states[tbd_envs, 0:3]
            self.heading_targets[tbd_envs, :] = wrap_to_pi(_target_heading + torch.atan2(pos_diff[:, 1:2], pos_diff[:, 0:1]))

            # For compatibility: still compute commands (used by some reward functions)
            self.commands[tbd_envs, :2] = quat_rotate_inverse(yaw_quat(self.base_quat[tbd_envs]), pos_diff)[:, :2]
            forward = quat_apply(self.base_quat[tbd_envs], self.forward_vec[tbd_envs])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[tbd_envs, 2] = wrap_to_pi(self.heading_targets[tbd_envs, 0] - heading)

            # Reject sampling (same logic as original)
            max_speed = self.cfg.commands.max_speed
            landing_time_min = self.cfg.commands.landing_time_min
            landing_time_max = self.cfg.commands.landing_time_max
            dist_xy = torch.norm(pos_diff[:, :2], dim=1)
            raw_time = dist_xy / max_speed
            min_valid_time = torch.clamp(raw_time, min=landing_time_min, max=landing_time_max)
            in_obst = torch.zeros_like(self.heading_targets[:, 0], dtype=torch.bool)
            in_obst[tbd_envs] = torch.logical_or(in_obst[tbd_envs], min_valid_time > landing_time_max)
            low = min_valid_time
            high = torch.full_like(low, landing_time_max)
            landing_time = low + (high - low) * torch.rand_like(low)
            self.timer_left[env_ids] = landing_time.flatten()
            tbd_envs = in_obst.nonzero(as_tuple=False).flatten()

        # ---- Compute ballistic launch parameters for the ball ----
        self._setup_ball_trajectory(env_ids)

    def _setup_ball_trajectory(self, env_ids):
        """Compute ball launch position and velocity for a ballistic trajectory
        that lands at position_targets after ball_flight_time seconds.
        """
        n = len(env_ids)
        if n == 0:
            return
        
        g = 9.81
        landing_pos = self.position_targets[env_ids]  # (n, 3) — where ball lands

        # Ball flight time = timer_left (time the robot has to get there)
        flight_time = self.timer_left[env_ids].clone()  # (n,)
        # Clamp to avoid division by zero
        flight_time = torch.clamp(flight_time, min=0.5)
        self.ball_flight_time[env_ids] = flight_time

        # Launch position: random offset behind the robot, elevated
        # Ball comes from 3-8m away, 2-5m height
        launch_dist = torch_rand_float(3.0, 8.0, (n, 1), device=self.device).squeeze(-1)
        launch_angle = torch_rand_float(-np.pi, np.pi, (n, 1), device=self.device).squeeze(-1)
        launch_height = torch_rand_float(2.0, 5.0, (n, 1), device=self.device).squeeze(-1)

        launch_pos = torch.zeros(n, 3, device=self.device)
        launch_pos[:, 0] = landing_pos[:, 0] + launch_dist * torch.cos(launch_angle)
        launch_pos[:, 1] = landing_pos[:, 1] + launch_dist * torch.sin(launch_angle)
        launch_pos[:, 2] = landing_pos[:, 2] + launch_height

        self.ball_launch_pos[env_ids] = launch_pos

        # Compute required launch velocity for ballistic trajectory:
        # landing_pos = launch_pos + v0 * T + 0.5 * a * T^2
        # => v0 = (landing_pos - launch_pos - 0.5 * a * T^2) / T
        T = flight_time
        launch_vel = torch.zeros(n, 3, device=self.device)
        launch_vel[:, 0] = (landing_pos[:, 0] - launch_pos[:, 0]) / T
        launch_vel[:, 1] = (landing_pos[:, 1] - launch_pos[:, 1]) / T
        # z: landing_z = launch_z + vz0*T - 0.5*g*T^2
        # => vz0 = (landing_z - launch_z + 0.5*g*T^2) / T
        launch_vel[:, 2] = (landing_pos[:, 2] - launch_pos[:, 2] + 0.5 * g * T * T) / T

        self.ball_launch_vel[env_ids] = launch_vel
        self.ball_time_elapsed[env_ids] = 0.0

        # Initialize ball position at launch
        self.ball_pos[env_ids] = launch_pos.clone()
        self.ball_vel[env_ids] = launch_vel.clone()

    def _update_ball_state(self):
        """Update simulated ball position and velocity along ballistic trajectory."""
        g = 9.81
        t = self.ball_time_elapsed  # (num_envs,)

        # Position: x(t) = x0 + vx0*t, y(t) = y0 + vy0*t, z(t) = z0 + vz0*t - 0.5*g*t^2
        self.ball_pos[:, 0] = self.ball_launch_pos[:, 0] + self.ball_launch_vel[:, 0] * t
        self.ball_pos[:, 1] = self.ball_launch_pos[:, 1] + self.ball_launch_vel[:, 1] * t
        self.ball_pos[:, 2] = self.ball_launch_pos[:, 2] + self.ball_launch_vel[:, 2] * t - 0.5 * g * t * t
        # Clamp z to ground level (ball doesn't go underground)
        self.ball_pos[:, 2] = torch.clamp(self.ball_pos[:, 2], min=0.0)

        # Velocity: vx(t) = vx0, vy(t) = vy0, vz(t) = vz0 - g*t
        self.ball_vel[:, 0] = self.ball_launch_vel[:, 0]
        self.ball_vel[:, 1] = self.ball_launch_vel[:, 1]
        self.ball_vel[:, 2] = self.ball_launch_vel[:, 2] - g * t

        # Advance time
        self.ball_time_elapsed += self.dt

    def _compute_ball_obs(self):
        """Compute noisy ball observations in robot body frame."""
        # Ball position relative to robot (world frame)
        ball_rel_pos = self.ball_pos - self.root_states[:, 0:3]
        # Ball velocity in world frame (already computed)
        ball_vel_world = self.ball_vel.clone()

        # Transform to robot body frame
        ball_pos_body = quat_rotate_inverse(self.base_quat, ball_rel_pos)
        ball_vel_body = quat_rotate_inverse(self.base_quat, ball_vel_world)

        # Add observation noise (simulating YOLO detection noise)
        pos_noise_std = getattr(self.cfg.noise.noise_scales, 'ball_pos', 0.05)
        vel_noise_std = getattr(self.cfg.noise.noise_scales, 'ball_vel', 0.3)

        if getattr(self, 'add_noise', False):
            ball_pos_body += torch.randn_like(ball_pos_body) * pos_noise_std * self.cfg.noise.noise_level
            ball_vel_body += torch.randn_like(ball_vel_body) * vel_noise_std * self.cfg.noise.noise_level

        self.ball_pos_body_noisy = ball_pos_body
        self.ball_vel_body_noisy = ball_vel_body

    def _post_physics_step_callback(self):
        """Override: update ball state + original callback logic (minus timer_left)."""
        # Still decrement timer_left for compatibility with some reward functions
        self.timer_left -= self.dt

        # Update ball trajectory
        self._update_ball_state()
        self._compute_ball_obs()

        # Update commands for position tracking rewards (same as original)
        pos_diff = self.position_targets - self.root_states[:, 0:3]
        self.commands[:, :2] = quat_rotate_inverse(yaw_quat(self.base_quat[:]), pos_diff)[:, :2]

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = wrap_to_pi(self.heading_targets[:, 0] - heading)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def compute_proprioceptive_observations(self):
        """E2E observations: replace timer_left + commands with raw ball state."""
        self.proprioceptive_obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,    # 3 (ωx, ωy, ωz)
            self.projected_gravity,                          # 3 (gx, gy, gz)
            self.ball_pos_body_noisy,                        # 3 ball pos in body frame (replaces timer_left)
            self.ball_vel_body_noisy,                        # 3 ball vel in body frame (replaces commands part 1)
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12 joint diff
            self.dof_vel * self.obs_scales.dof_vel,          # 12 joint velocity
            self.actions,                                     # 12 last actions
        ), dim=-1)  # Total: 3+3+3+3+12+12+12 = 48
        return self.proprioceptive_obs_buf

    def compute_observations(self):
        """E2E observations: no privileged obs, no teacher-student."""
        self.obs_buf = self.compute_proprioceptive_observations()

        # Add noise (for proprioceptive parts only, ball noise already added)
        if getattr(self, 'add_noise', False):
            noise_vec = self._get_noise_scale_vec(self.cfg)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * noise_vec

        # NO privileged observations for E2E baseline
        # Critic uses same obs as actor (no information advantage)
        if self.cfg.env.num_privileged_obs is not None:
            # Use same obs for critic, but without noise
            self.privileged_obs_buf = self.obs_buf.clone()

        self.obs_history_buf = torch.cat((self.obs_history_buf[:, self.obs_buf.shape[1]:], self.obs_buf), dim=1)

    def _get_noise_scale_vec(self, cfg):
        """Noise vector adapted for E2E observation structure (48-dim)."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level                             # gravity
        noise_vec[6:9] = 0.  # ball_pos (noise already applied in _compute_ball_obs)
        noise_vec[9:12] = 0.  # ball_vel (noise already applied in _compute_ball_obs)
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dof_vel
        noise_vec[36:48] = 0.  # previous actions
        return noise_vec

    # ==================== E2E Reward Functions ====================
    # These replace the timer-gated rewards with continuous versions.

    def _reward_track_ball_landing(self):
        """Continuous reward for moving toward the ball's landing position.
        No timer gating — always active. The policy must learn WHEN to move."""
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        return torch.exp(-distance / 1.0)  # Exponential shaping, sigma=1.0

    def _reward_reach_landing_pos(self):
        """Dense reward for being close to the landing position."""
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        return 1.0 / (1.0 + torch.square(distance / self.cfg.rewards.position_target_sigma_tight))

    def _reward_catch_bonus(self):
        """Sparse bonus for being at the landing position when the ball arrives."""
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        ball_landed = self.ball_time_elapsed >= self.ball_flight_time
        close_enough = distance < self.cfg.rewards.position_target_sigma_tight
        return (ball_landed & close_enough).float()

    # Override timer-gated rewards to be continuous for E2E
    def _reward_task(self):
        """Task reward WITHOUT timer-conditioned masking (E2E version)."""
        dist2 = torch.square(torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1))
        return 1.0 / (1.0 + dist2)

    def _reward_reach_pos_target_tight(self):
        """Position target reward WITHOUT timer gating (E2E version)."""
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        return 1.0 / (1.0 + torch.square(distance / self.cfg.rewards.position_target_sigma_tight))

    def _reward_termination(self):
        """Termination reward WITHOUT timer gating (E2E version)."""
        return (self.reset_buf * ~self.time_out_buf).float()

    def _reward_stand_still_pos(self):
        """Stand still reward WITHOUT timer gating (E2E version).
        Penalize motion when already at the target."""
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        stand_bias = torch.zeros_like(self.dof_pos)
        stand_bias[:, 1::3] += 0.2
        stand_bias[:, 2::3] -= 0.3
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos - stand_bias), dim=1) * \
               (distance < self.cfg.rewards.position_target_sigma_tight).float()
