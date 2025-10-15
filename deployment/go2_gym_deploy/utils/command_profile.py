import torch
import numpy as np
import math

class CommandProfile:
    def __init__(self, dt, max_time_s=10.):
        self.dt = dt
        self.max_timestep = int(max_time_s / self.dt)
        self.commands = torch.zeros((self.max_timestep, 9))
        self.start_time = 0

    def get_command(self, t):
        timestep = int((t - self.start_time) / self.dt)
        timestep = min(timestep, self.max_timestep - 1)
        return self.commands[timestep, :]

    def get_buttons(self):
        return [0, 0, 0, 0]

    def reset(self, reset_time):
        self.start_time = reset_time


class ConstantAccelerationProfile(CommandProfile):
    def __init__(self, dt, max_speed, accel_time, zero_buf_time=0):
        super().__init__(dt)
        zero_buf_timesteps = int(zero_buf_time / self.dt)
        accel_timesteps = int(accel_time / self.dt)
        self.commands[:zero_buf_timesteps] = 0
        self.commands[zero_buf_timesteps:zero_buf_timesteps + accel_timesteps, 0] = torch.arange(0, max_speed,
                                                                                                 step=max_speed / accel_timesteps)
        self.commands[zero_buf_timesteps + accel_timesteps:, 0] = max_speed


class ElegantForwardProfile(CommandProfile):
    def __init__(self, dt, max_speed, accel_time, duration, deaccel_time, zero_buf_time=0):
        import numpy as np

        zero_buf_timesteps = int(zero_buf_time / dt)
        accel_timesteps = int(accel_time / dt)
        duration_timesteps = int(duration / dt)
        deaccel_timesteps = int(deaccel_time / dt)

        total_time_s = zero_buf_time + accel_time + duration + deaccel_time

        super().__init__(dt, total_time_s)

        x_vel_cmds = [0] * zero_buf_timesteps + [*np.linspace(0, max_speed, accel_timesteps)] + \
                     [max_speed] * duration_timesteps + [*np.linspace(max_speed, 0, deaccel_timesteps)]

        self.commands[:len(x_vel_cmds), 0] = torch.Tensor(x_vel_cmds)


class ElegantYawProfile(CommandProfile):
    def __init__(self, dt, max_speed, zero_buf_time, accel_time, duration, deaccel_time, yaw_rate):
        import numpy as np

        zero_buf_timesteps = int(zero_buf_time / dt)
        accel_timesteps = int(accel_time / dt)
        duration_timesteps = int(duration / dt)
        deaccel_timesteps = int(deaccel_time / dt)

        total_time_s = zero_buf_time + accel_time + duration + deaccel_time

        super().__init__(dt, total_time_s)

        x_vel_cmds = [0] * zero_buf_timesteps + [*np.linspace(0, max_speed, accel_timesteps)] + \
                     [max_speed] * duration_timesteps + [*np.linspace(max_speed, 0, deaccel_timesteps)]

        yaw_vel_cmds = [0] * zero_buf_timesteps + [0] * accel_timesteps + \
                       [yaw_rate] * duration_timesteps + [0] * deaccel_timesteps

        self.commands[:len(x_vel_cmds), 0] = torch.Tensor(x_vel_cmds)
        self.commands[:len(yaw_vel_cmds), 2] = torch.Tensor(yaw_vel_cmds)


class ElegantGaitProfile(CommandProfile):
    def __init__(self, dt, filename):
        import numpy as np
        import json

        with open(f'../command_profiles/{filename}', 'r') as file:
                command_sequence = json.load(file)

        len_command_sequence = len(command_sequence["x_vel_cmd"])
        total_time_s = int(len_command_sequence / dt)

        super().__init__(dt, total_time_s)

        self.commands[:len_command_sequence, 0] = torch.Tensor(command_sequence["x_vel_cmd"])
        self.commands[:len_command_sequence, 2] = torch.Tensor(command_sequence["yaw_vel_cmd"])
        self.commands[:len_command_sequence, 3] = torch.Tensor(command_sequence["height_cmd"])
        self.commands[:len_command_sequence, 4] = torch.Tensor(command_sequence["frequency_cmd"])
        self.commands[:len_command_sequence, 5] = torch.Tensor(command_sequence["offset_cmd"])
        self.commands[:len_command_sequence, 6] = torch.Tensor(command_sequence["phase_cmd"])
        self.commands[:len_command_sequence, 7] = torch.Tensor(command_sequence["bound_cmd"])
        self.commands[:len_command_sequence, 8] = torch.Tensor(command_sequence["duration_cmd"])

def wrap_to_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

class RCControllerProfile(CommandProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0,
                 max_speed=0.5, max_yaw_rate=0.6/2, joystick_meter_per_second=1.5):
        """
        joystick_meter_per_second: When the joystick is fully deflected (1.0), target_pos moves in meters per second (for moving target_pos with the joystick)
        max_speed: The upper limit of the robot's uniform speed (m/s), used to calculate target_time = dist / max_speed
        """
        super().__init__(dt)
        self.state_estimator = state_estimator
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale

        self.probe_vel_multiplier = 1.0

        # triggerable commands (as before)
        self.triggered_commands = {i: None for i in range(4)}
        self.currently_triggered = [0, 0, 0, 0]
        self.button_states = [0, 0, 0, 0]

        # --- new fields for target-position control ---
        self.target_pos = None            # np.array([x,y,z]) or None until initialized
        self.last_valid_st_tgt = np.array([0.0, 0.0])
        self.move_active = False
        self.remaining_time = 0.0
        self.max_speed = 0.5
        self.max_yaw_rate = float(max_yaw_rate)
        self.joystick_meter_per_second = float(joystick_meter_per_second)
        # which button index corresponds to R2? (you must confirm this mapping in cheetah_state_estimator)
        self.R1_BUTTON_IDX = 3

    def get_command(self, t, probe=False):
        """
        Returns command: numpy array shape (9,) or at least length >= 3 where the first 3 are vx, vy, yaw_rate (unit: m/s, rad/s)
            Behavior:
            - Joystick movement: Update self.target_pos (x, y)
            - R2 not pressed: Return to zero velocity
            - R2 pressed (rising edge): Calculate remaining_time = distance / max_speed and start moving
            - While moving: Generate linear and angular velocities based on remaining_time, remaining_time -= self.dt
            - Arrival: Stop moving
        """
        reset_timer = False
        # read raw joystick "command" (legacy usage): assume it returns array-like [axis_x, axis_y, yaw_axis]
        raw_cmd = np.array(self.state_estimator.get_command())
        # read buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()

        # ensure target_pos initialized as current base pos
        base_yaw = np.array([0.0])
        dog_coord = self.state_estimator.get_dog_coord()
        if self.target_pos is None:
            self.target_pos = np.array([0.0, 0.0])

        # --- joystick: move target_pos in x,y in robot body frame ---
        # raw_cmd[0], raw_cmd[1] are joystick deflections [-1..1]
        if self.target_pos is None:
            # fallback: if we don't have base pos yet, do nothing to target
            pass
        else:
            # dt exists from CommandProfile
            dx = raw_cmd[0] * self.joystick_meter_per_second * self.dt
            dy = raw_cmd[1] * -self.joystick_meter_per_second * self.dt # to account for joystick y being inverted
            # note: assume joystick axes are in robot body frame; if they are world-frame you'd need transform
            self.target_pos[0] += dx
            self.target_pos[1] += dy
            if not self.move_active:
                # print(f"target:{self.target_pos[0]:.2f},{self.target_pos[1]:.2f}")
                # print(f"currpos:{dog_coord[0]:.2f},{dog_coord[1]:.2f}")
                pass
        if self.state_estimator.get_T_world_to_dog() is not None:
            self.T_world_to_dog = self.state_estimator.get_T_world_to_dog()
        else:
            self.T_world_to_dog = np.eye(4)  # fallback
        # --- handle R2 press to start movement (rising edge triggers) ---
        r1_pressed = (self.button_states[self.R1_BUTTON_IDX] == 1)
        r1_prev = (prev_button_states[self.R1_BUTTON_IDX] == 1)
        if r1_pressed and not r1_prev:
            self.move_active = not self.move_active
            # rising edge -> start moving towards target
            if not self.move_active:
                reset_timer = True

        # --- produce velocity command depending on move_active ---
        cmd = np.zeros(9, dtype=float)  # keep same shape as other profiles (first 3 are vx,vy,yaw)
        
        # auxiliary rotation limit parameters
        tangent_aux_dist = 2.35
        max_yaw_rad = np.pi/3
        aux_rad = np.pi*2/3
        if not self.move_active:
            self.state_estimator.publish_dog_target(self.target_pos, [0.0, 0.0])
        # shift target pos to shuttlecock if valid
        shuttlecock_target, target_valid = self.state_estimator.get_shuttlecock_target()
        if target_valid:
            self.last_valid_st_tgt = shuttlecock_target
            print(f"update tgt:({self.last_valid_st_tgt[0]:.2f}, {self.last_valid_st_tgt[1]:.2f})")
        if self.move_active and self.last_valid_st_tgt is not None and target_valid == 1:
            vec_world = self.last_valid_st_tgt - dog_coord[0:2]
            dist = float(np.linalg.norm(vec_world))
            self.remaining_time = float(dist / max(self.max_speed, 1e-3))
            if dist < 0.15:  # threshold reached
                self.remaining_time = 0.0
                cmd[0] = 0.0
                cmd[1] = 0.0
                cmd[2] = 0.0
            else:
                dx_world = self.last_valid_st_tgt[0] - dog_coord[0] # x position command
                dy_world = self.last_valid_st_tgt[1] - dog_coord[1]  # y position command
                tmp_vec_in_frame_dog = self.T_world_to_dog@np.array([dx_world, dy_world, 0, 0])
                yaw_diff = np.arctan2(tmp_vec_in_frame_dog[1], tmp_vec_in_frame_dog[0])
                if abs(yaw_diff) > max_yaw_rad:
                    sign = 1 if yaw_diff > 0 else -1
                    aux_x = tangent_aux_dist * np.cos(sign * aux_rad)
                    aux_y = tangent_aux_dist * np.sin(sign * aux_rad)
                    cmd[0] = aux_x
                    cmd[1] = aux_y
                    cmd[2] = np.clip(sign * aux_rad, -0.3, 0.3) # 17.2deg
                else:
                    cmd[0] = tmp_vec_in_frame_dog[0]
                    cmd[1] = tmp_vec_in_frame_dog[1]
                    cmd[2] = np.clip(yaw_diff, -0.3, 0.3)

                # # baseline
                # kp = 2.0
                # cmd[0] = tmp_vec_in_frame_dog[0] * kp
                # cmd[1] = tmp_vec_in_frame_dog[1] * kp
                # cmd[2] = np.clip(kp * yaw_diff, -0.5, 0.5)

                print(f"World_v:({dx_world:.2f}, {dy_world:.2f}), cmd_actual:({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]/np.pi*180:.2f}deg), time:{self.remaining_time*1e3:.2f}ms")
                self.state_estimator.publish_dog_target(self.target_pos, cmd[0:2])

        else:
            # not moving -> zero
            cmd[0] = 0.0; cmd[1] = 0.0; cmd[2] = 0.0

        # apply probe multiplier or scaling if required
        if probe:
            cmd[0] *= self.probe_vel_multiplier
            cmd[2] *= self.probe_vel_multiplier

        return cmd, reset_timer, self.remaining_time


    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()

class RCControllerProfileAccel(RCControllerProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0):
        super().__init__(dt, state_estimator, x_scale=x_scale, y_scale=y_scale, yaw_scale=yaw_scale)
        self.position_command = torch.zeros(3)

    def get_command(self, t):
        pos_command = self.state_estimator.get_command()
        self.position_command[0] = pos_command[0] * self.x_scale
        self.position_command[1] = pos_command[1] * self.y_scale
        self.position_command[2] = pos_command[2] * self.yaw_scale

        # check for action buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()
        for button in range(4):
            if self.button_states[button] == 1 and self.triggered_commands[button] is not None:
                if prev_button_states[button] == 0:
                    # reset the triggered action
                    self.triggered_commands[button].reset(t)
                # execute the triggered action
                return self.triggered_commands[button].get_command(t)

        return self.velocity_command[:]

    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()

class KeyboardProfile(CommandProfile):
    # for control via keyboard inputs to isaac gym visualizer
    def __init__(self, dt, isaac_env, x_scale=1.0, y_scale=1.0, yaw_scale=1.0):
        super().__init__(dt)
        from isaacgym.gymapi import KeyboardInput
        self.gym = isaac_env.gym
        self.viewer = isaac_env.viewer
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_UP, "FORWARD")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_DOWN, "REVERSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_LEFT, "LEFT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_RIGHT, "RIGHT")

        self.keyb_command = [0, 0, 0]
        self.command = [0, 0, 0]

    def get_command(self, t):
        events = self.gym.query_viewer_action_events(self.viewer)
        events_dict = {event.action: event.value for event in events}
        # print(events_dict)
        if "FORWARD" in events_dict and events_dict["FORWARD"] == 1.0: self.keyb_command[0] = 1.0
        if "FORWARD" in events_dict and events_dict["FORWARD"] == 0.0: self.keyb_command[0] = 0.0
        if "REVERSE" in events_dict and events_dict["REVERSE"] == 1.0: self.keyb_command[0] = -1.0
        if "REVERSE" in events_dict and events_dict["REVERSE"] == 0.0: self.keyb_command[0] = 0.0
        if "LEFT" in events_dict and events_dict["LEFT"] == 1.0: self.keyb_command[1] = 1.0
        if "LEFT" in events_dict and events_dict["LEFT"] == 0.0: self.keyb_command[1] = 0.0
        if "RIGHT" in events_dict and events_dict["RIGHT"] == 1.0: self.keyb_command[1] = -1.0
        if "RIGHT" in events_dict and events_dict["RIGHT"] == 0.0: self.keyb_command[1] = 0.0

        self.command[0] = self.keyb_command[0] * self.x_scale
        self.command[1] = self.keyb_command[2] * self.y_scale
        self.command[2] = self.keyb_command[1] * self.yaw_scale

        # print(self.command)

        return self.command


if __name__ == "__main__":
    cmdprof = ConstantAccelerationProfile(dt=0.2, max_speed=4, accel_time=3)
    # print(cmdprof.commands)
    # print(cmdprof.get_command(2))
