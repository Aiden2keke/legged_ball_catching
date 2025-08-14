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

# class RCControllerProfile(CommandProfile):
#     def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0, probe_vel_multiplier=1.0):
#         super().__init__(dt)
#         self.state_estimator = state_estimator
#         self.x_scale = x_scale
#         self.y_scale = y_scale
#         self.yaw_scale = yaw_scale

#         self.probe_vel_multiplier = probe_vel_multiplier

#         self.triggered_commands = {i: None for i in range(4)}  # command profiles for each action button on the controller
#         self.currently_triggered = [0, 0, 0, 0]
#         self.button_states = [0, 0, 0, 0]

#     def get_command(self, t, probe=False):

#         command = self.state_estimator.get_command()
#         command[0] = command[0] * self.x_scale
#         command[1] = command[1] * self.y_scale
#         command[2] = command[2] * self.yaw_scale

#         reset_timer = False

#         if probe:
#             command[0] = command[0] * self.probe_vel_multiplier
#             command[2] = command[2] * self.probe_vel_multiplier

#         # check for action buttons
#         prev_button_states = self.button_states[:]
#         self.button_states = self.state_estimator.get_buttons()
#         for button in range(4):
#             if self.triggered_commands[button] is not None:
#                 if self.button_states[button] == 1 and prev_button_states[button] == 0:
#                     if not self.currently_triggered[button]:
#                         # reset the triggered action
#                         self.triggered_commands[button].reset(t)
#                         # reset the internal timing variable
#                         reset_timer = True
#                         self.currently_triggered[button] = True
#                     else:
#                         self.currently_triggered[button] = False
#                 # execute the triggered action
#                 if self.currently_triggered[button] and t < self.triggered_commands[button].max_timestep:
#                     command = self.triggered_commands[button].get_command(t)


#         return command, reset_timer

def wrap_to_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

class RCControllerProfile(CommandProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0,
                 max_speed=0.7, max_yaw_rate=0.6, joystick_meter_per_second=0.5):
        """
        joystick_meter_per_second: 当摇杆满偏 (1.0) 时，target_pos 每秒移动多少米（用于用摇杆移动 target_pos）
        max_speed: 机器人匀速上限（m/s），用于计算 target_time = dist / max_speed
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
        self.move_active = False
        self.remaining_time = 0.0
        self.max_speed = 0.5
        self.max_yaw_rate = float(max_yaw_rate)
        self.joystick_meter_per_second = float(joystick_meter_per_second)
        # which button index corresponds to R2? (you must confirm this mapping in cheetah_state_estimator)
        self.R2_BUTTON_IDX = 2  # <-- **请根据你的 cheetah_state_estimator.py 对应映射修改此值**

    def get_command(self, t, probe=False):
        """
        返回 command: numpy array shape (9,) or at least length>=3 where first 3 are vx,vy,yaw_rate (unit: m/s, rad/s)
        Behavior:
         - 摇杆移动：更新 self.target_pos（x,y）
         - 未按 R2：返回零速度
         - 按 R2（升沿）：计算 remaining_time = distance / max_speed 并开始移动
         - 移动中：根据 remaining_time 生成线速度和角速度，remaining_time -= self.dt
         - 到达：停止移动
        """
        reset_timer = False
        # read raw joystick "command" (legacy usage): assume it returns array-like [axis_x, axis_y, yaw_axis]
        raw_cmd = np.array(self.state_estimator.get_command())  # e.g. [-1..1] or velocities depending on estimator
        # read buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()

        # ensure target_pos initialized as current base pos
        base_pos = np.array([0, 0, 0])
        base_yaw = np.array([0])
        if self.target_pos is None and base_pos is not None:
            self.target_pos = base_pos.copy()

        # --- joystick: move target_pos in x,y in robot body frame ---
        # raw_cmd[0], raw_cmd[1] are joystick deflections [-1..1]
        if self.target_pos is None:
            # fallback: if we don't have base pos yet, do nothing to target
            pass
        else:
            # dt exists from CommandProfile
            dx = raw_cmd[0] * self.joystick_meter_per_second * self.dt
            dy = raw_cmd[1] * self.joystick_meter_per_second * self.dt
            # note: assume joystick axes are in robot body frame; if they are world-frame you'd need transform
            self.target_pos[0] += dx
            self.target_pos[1] += dy

        # --- handle R2 press to start movement (rising edge triggers) ---
        r2_pressed = (self.button_states[self.R2_BUTTON_IDX] == 1)
        r2_prev = (prev_button_states[self.R2_BUTTON_IDX] == 1)
        if r2_pressed and not r2_prev and not self.move_active:
            # rising edge -> start moving towards target
            if base_pos is not None and self.target_pos is not None:
                dist = float(np.linalg.norm(self.target_pos[:2] - base_pos[:2]))
                if dist < 1e-4:
                    self.move_active = False
                    self.remaining_time = 0.0
                else:
                    self.move_active = True
                    self.remaining_time = float(dist / max(self.max_speed, 1e-3))
                    reset_timer = True

        # if some triggered_commands exist we preserve that logic (execute triggered commands)
        # (this mirrors previous behavior)
        for button in range(4):
            if self.triggered_commands[button] is not None:
                # keep previous triggered-logic
                if self.button_states[button] == 1 and prev_button_states[button] == 0:
                    if not self.currently_triggered[button]:
                        self.triggered_commands[button].reset(t)
                        reset_timer = True
                        self.currently_triggered[button] = True
                    else:
                        self.currently_triggered[button] = False
                if self.currently_triggered[button] and t < self.triggered_commands[button].max_timestep:
                    return self.triggered_commands[button].get_command(t), reset_timer

        # --- produce velocity command depending on move_active ---
        cmd = np.zeros(9, dtype=float)  # keep same shape as other profiles (first 3 are vx,vy,yaw)
        if self.move_active and base_pos is not None:
            # compute remaining_time based velocity (simple proportional)
            # linear velocity toward target in body frame. We need vector from base->target in body frame.
            # Here we assume target_pos is expressed in world frame and base_pos is world frame.
            vec_world = self.target_pos[:2] - base_pos[:2]  # [dx,dy] in world
            dist = float(np.linalg.norm(vec_world))
            if dist < 0.02:  # threshold reached
                self.move_active = False
                self.remaining_time = 0.0
                # keep cmd zero
            else:
                # convert displacement to body frame using base_yaw if available
                if base_yaw is not None:
                    c = math.cos(-base_yaw); s = math.sin(-base_yaw)
                    # rotation by -yaw: body = R(-yaw) * world_vec
                    body_dx = c * vec_world[0] - s * vec_world[1]
                    body_dy = s * vec_world[0] + c * vec_world[1]
                else:
                    # if no yaw info, treat world==body (best-effort)
                    body_dx = vec_world[0]; body_dy = vec_world[1]

                # desired linear velocity (approx): v = displacement / remaining_time
                denom = max(self.remaining_time, self.dt, 1e-3)
                vx_des = body_dx / denom
                vy_des = body_dy / denom
                # clip by max_speed (in xy plane)
                v_norm = math.hypot(vx_des, vy_des)
                if v_norm > self.max_speed:
                    vx_des = vx_des / v_norm * self.max_speed
                    vy_des = vy_des / v_norm * self.max_speed

                # desired yaw: rotate robot to face the target point (optional)
                # compute heading target in world frame and current heading -> yaw_error
                if base_yaw is not None:
                    target_heading = math.atan2(vec_world[1], vec_world[0])
                    yaw_err = wrap_to_pi(target_heading - base_yaw)
                    yaw_rate_des = yaw_err / denom
                    # clip yaw_rate
                    yaw_rate_des = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate_des))
                else:
                    yaw_rate_des = 0.0

                cmd[0] = vx_des
                cmd[1] = vy_des
                cmd[2] = yaw_rate_des

                # decrement remaining_time
                self.remaining_time = max(0.0, self.remaining_time - self.dt)
                if self.remaining_time <= 0.0:
                    # reached end of planned time -> stop
                    self.move_active = False

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
        # self.x_scale, self.y_scale, self.yaw_scale = self.x_scale / 100., self.y_scale / 100., self.yaw_scale / 100.
        # self.velocity_command = torch.zeros(3)
        
        # 将速度指令改为位置指令
        self.position_command = torch.zeros(3)

    def get_command(self, t):

        # accel_command = self.state_estimator.get_command()
        # self.velocity_command[0] = self.velocity_command[0]  + accel_command[0] * self.x_scale
        # self.velocity_command[1] = self.velocity_command[1]  + accel_command[1] * self.y_scale
        # self.velocity_command[2] = self.velocity_command[2]  + accel_command[2] * self.yaw_scale

        # 将速度指令改为位置指令
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
