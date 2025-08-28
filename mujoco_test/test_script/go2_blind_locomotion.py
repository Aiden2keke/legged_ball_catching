#!/usr/bin/env python3

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
unitree_mujoco_test_dir = os.path.join(current_dir, "..")
sys.path.append(unitree_mujoco_test_dir)
actor_dir = os.path.join(unitree_mujoco_test_dir, "model/rsl_rl_teacher_student/actor")
proprio_encoder_dir = os.path.join(unitree_mujoco_test_dir, "model/rsl_rl_teacher_student/proprio_encoder")
robot_data_dir = os.path.join(unitree_mujoco_test_dir, "data")

import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from module.modules import Actor, MLPEncoder
from pynput import keyboard
import math


def quat_rotate_inverse(q, v):
    # 假设 q 的形状为 (4,)，v 的形状为 (3,)
    q_w = q[0]
    q_vec = q[1:]
    # 计算部分 a
    a = v * (2.0 * q_w**2 - 1.0)
    # 计算部分 b
    b = np.cross(q_vec, v) * q_w * 2.0
    # 计算部分 c
    c = q_vec * (np.dot(q_vec, v) * 2.0)

    return a - b + c

# # 带归一化的“逆旋转”函数
# def quat_rotate_inverse(q, v):
#     # 确保 q 归一化
#     q = q / np.linalg.norm(q)
#     w, x, y, z = q
#     q_vec = np.array([x, y, z])

#     a = v * (2*w*w - 1)
#     b = np.cross(q_vec, v) * (2*w)
#     c = q_vec * (2 * np.dot(q_vec, v))

#     # 逆旋转公式：a - b + c
#     return a - b + c


##############
# 初始化键盘监听
##############
# 全局变量，保存键盘状态
key_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "l": False,
    "r": False,
}
# 最大和最小速度限制
max_speed = 0.5
min_speed = -0.5
# 阻尼系数，用于速度平滑下降
damping_factor = 0.2  # 调整该值以改变速度衰减快慢


def set_target_by_key(key):
    target_pos = np.zeros(3)
    target_time = 3.0
    a = 0.5 / 10
    # 获取机器人朝向（世界坐标系下）
    # fw = quat_rotate_inverse(body_quat, np.array([1, 0, 0]))  # 前方
    # left = quat_rotate_inverse(body_quat, np.array([0, 1, 0]))  # 左方
    fw=np.array([1, 0, 0])
    left=np.array([0, 1, 0])
    if key == "up":
        return fw * a, target_time
    elif key == "down":
        return -fw * a, target_time
    elif key == "left":
        return left * a, target_time
    elif key == "right":
        return -left * a, target_time
    else:
        return target_pos, target_time

def on_press(key):
    """按键按下时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = True
    elif key == keyboard.Key.down:
        key_state["down"] = True
    elif key == keyboard.Key.left:
        key_state["left"] = True
    elif key == keyboard.Key.right:
        key_state["right"] = True
    elif key == keyboard.KeyCode.from_char("r"):
        key_state["l"] = True
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = True


def on_release(key):
    """按键释放时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = False
    elif key == keyboard.Key.down:
        key_state["down"] = False
    elif key == keyboard.Key.left:
        key_state["left"] = False
    elif key == keyboard.Key.right:
        key_state["right"] = False
    elif key == keyboard.KeyCode.from_char("r"):
        key_state["l"] = False
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = False


##############
# 加载机器人模型
##############
model = mujoco.MjModel.from_xml_path(robot_data_dir + "/go2/scene_terrain.xml")
data = mujoco.MjData(model)

############################
# 加载 policy 和 encoder 模型
############################
device = torch.device("cuda")
actor = Actor(num_obs=49, num_actions=12, hidden_dims=[512, 256, 128])
# actor.load_state_dict(torch.load(actor_dir + "/actor_0910_cosine_reinforce.pth"))
actor.load_state_dict(torch.load(actor_dir + "/actor_oracle41-4-7500.pth"))
actor = actor.to(device)
actor.eval()
# proprio_encoder = MLPEncoder(input_dim=45*15)
# proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + "/proprio_encoder_0910_cosine_reinforce.pth"))
# proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + "/proprio_oracle.pth"))
# proprio_encoder = proprio_encoder.to(device)
# proprio_encoder.eval()

###################
# 初始化储存状态的变量
###################
global command
body_pos = np.zeros(3)  # 机器人的位置 在世界坐标系下
body_quat = np.zeros(4)  # 机器人的orientation 在世界坐标系下
body_lin_vel = np.zeros(3)
body_ang_vel = np.zeros(3)
gravity_projection = np.zeros(3)
command = np.zeros(3)
joint_pos = np.zeros(12)
joint_vel = np.zeros(12)
last_action = np.zeros(12)
torques = np.zeros(12)
default_joint_angles = {  # = target angles [rad] when action = 0.0
    "FL_hip_joint": 0.1,  # [rad]
    "FL_thigh_joint": 0.8,  # [rad]
    "FL_calf_joint": -1.5,  # [rad]

    "FR_hip_joint": -0.1,  # [rad]
    "FR_thigh_joint": 0.8,  # [rad]
    "FR_calf_joint": -1.5,  # [rad]

    "RL_hip_joint": 0.1,  # [rad]
    "RL_thigh_joint": 1.0,  # [rad]
    "RL_calf_joint": -1.5,  # [rad]

    "RR_hip_joint": -0.1,  # [rad]
    "RR_thigh_joint": 1.0,  # [rad]
    "RR_calf_joint": -1.5,  # [rad]
}
default_dof_pos = np.array(list(default_joint_angles.values()))
data.qpos[0:3] = np.array([0.0, 0.0, 0.37])  # 0.37 是合适的高度，防止机器人一开始就掉下去
target_pos = np.array([0.0, 0.0, 0])
target_time = 0.0
dt = 0.02  # 仿真步长（可根据实际设置）

#####################################
# 力矩计算相关增益以及 observation scale
#####################################
p_gains = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])  # 位置项增益
d_gains = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # 速度项增益
torque_limits = np.array([20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0])
actions_scale = 0.25
body_lin_vel_scale = 2.0
body_ang_vel_scale = 0.25
command_scale = np.array([2.0, 2.0, 0.25])
joint_vel_scale = 0.05
joint_pos_scale = 1.0
episode_length_s = 20
arget_active = False

##############
# 关节状态初始化
##############
data.qpos[7:19] = default_dof_pos

#########################
# 监听键盘输入以发送 command
#########################
command = np.array([0.0, 0.0, 0.0])
prev_command = np.array([0.0, 0.0, 0.0])
print(
    "Use arrow keys to move the robot.\n",
    # "'l': turn left\n",
    # "'r': turn left\n",
    "UP: move forward,\n",
    "DOWN: move backward,\n",
    "LEFT: move left,\n" "RIGHT: move right.\n",
)
# 监听键盘输入
command_scale_factor = 0.005
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
# proprio_obs_history = torch.zeros((1, 45*15)).to(device)

###########
# 仿真主循环
###########
m = model
d = data
velocity_scale_factor = 2.0  # 调整此值以使箭头长度在视觉上更明显
min_arrow_length = 0.05      # 即使速度为零，也确保箭头有一个最小可见长度
max_arrow_length = 1.0       # 限制箭头的最大长度，防止在高速时过长导致画面混乱
dist_history = torch.zeros(50).to(device)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = [-7., 4.5, 3.]  # Example: move camera to (0, 0, 0.5)
    viewer.cam.azimuth = -40 # Example: rotate camera by 180 degree
    viewer.cam.elevation = -30  # Example: tilt camera by 30 degree
    start = time.time()
    last_update_time = start
    t = 0.0
    # max_command_value = 0.5
    while viewer.is_running() and time.time() - start < 5000:
        # 获取当前机器人位置和朝向
        body_pos = np.array(data.qpos[0:3])
        body_quat = np.array(data.qpos[3:7])
        # target_pos = np.array([-3, 3, 0])/2
        t += dt

    # 检查按键并设置目标点和时间
        for direction in ["up", "down", "left", "right"]:
            if key_state[direction]:
                target_pos_add, target_time = set_target_by_key(direction)
                print("target_pos_add:", target_pos_add)
                target_pos = target_pos + target_pos_add
                arget_active = True
                break  # 只响应一个方向
    # 计算 command（期望速度），根据目标点和当前位置
        print("target_pos:", target_pos)
        print("body_pos:", body_pos)
        pos_diff = target_pos - body_pos
        print("pos_diff:", pos_diff)
        distance = np.linalg.norm(pos_diff[:2])
        print("distance:", distance)
        dist_history = torch.cat((dist_history[1:], torch.tensor([distance], device=dist_history.device)), dim=-1)
        arget_active = True
        if (dist_history < 0.1 / 2).all():
            arget_active = False
            distance = 0.0
        if arget_active:
            command[:2] = quat_rotate_inverse(body_quat, pos_diff)[:2] #转变到机器人坐标下
        # 朝向误差（假设目标朝向为目标点方向）
            heading = np.arctan2(pos_diff[1], pos_diff[0])
        # 当前朝向
            fw = quat_rotate_inverse(body_quat, np.array([1, 0, 0]))
            current_heading = np.arctan2(fw[1], fw[0])
            # command[2] = heading - current_heading
            command[2] = np.clip(heading - current_heading, -0.3, 0.3)
            if t == dt:
                target_time = distance / max_speed if distance > 0 else 0.0
            else:
                target_time -= dt
        else:
            command[:2] = [0.0, 0.0]
            command[2] = 0.0
            target_time = 0.0
        print("command:", command)
        print("target_time:", target_time)

        ############
        # 更新模型输入
        ############
        current_time = time.time()

        body_lin_vel = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array(data.qvel[0:3]))
        body_ang_vel = np.array(data.qvel[3:6])  # mujoco 中的角速度是在局部坐标系下的，所以不需要转换
        gravity_projection = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array([0, 0, -1.0]))
        command_input = command
        # command_input=np.array([1.0,0.0,0.0]) # 直接一直让vx=1
        joint_pos = np.array(data.qpos[7:19])
        joint_vel = np.array(data.qvel[6:18])

        # proprio_observation = np.concatenate(
        #     [
        #         body_ang_vel * body_ang_vel_scale,
        #         gravity_projection,
        #         command_input * command_scale,
        #         (joint_pos - default_dof_pos) * joint_pos_scale,
        #         joint_vel * joint_vel_scale,
        #         last_action,
        #     ]
        # )

        proprio_observation = np.concatenate(
            [
                body_lin_vel * body_lin_vel_scale,
                body_ang_vel * body_ang_vel_scale,
                gravity_projection,
                command_input,
                # command_input * command_scale,
                [target_time / episode_length_s],
                (joint_pos - default_dof_pos) * joint_pos_scale,
                joint_vel * joint_vel_scale,
                last_action,
            ]
        )

        # noise_strength = 0.02  # 噪声的强度，值越大，噪声越强
        # noise = np.random.randn(*proprio_observation.shape) * noise_strength  # 生成与 proprio_observation 形状相同的噪声
        # proprio_observation += noise  # 将噪声添加到 proprio_observation# 加入噪声

        proprio_observation = torch.from_numpy(proprio_observation).float().to(device).unsqueeze(0) # shape (1,45)
        # proprio_obs_history = torch.cat((proprio_obs_history[:,45:], proprio_observation),dim=-1) # shape (1,675)

        #####################################
        # 将 observation 输入模型得到输出 action
        #####################################
        # proprio_latent = proprio_encoder(proprio_obs_history)
        # proprio_latent = torch.nn.functional.normalize(proprio_latent, p=2, dim=-1)
        # actor_input = torch.cat((proprio_observation, proprio_latent), dim=-1)
        actor_input = proprio_observation

        action = actor(actor_input)
        action = action.detach().cpu().numpy()
        action = np.clip(action, -6.0, 6.0)
        last_action[:] = action[:]

        ####################################
        # 将 action 映射为 torque 作为控制输出
        ####################################
        decimation = 4
        for i in range(decimation):
            # 计算力矩
            dof_pos = np.array(data.qpos[7:19])
            dof_vel = np.array(data.qvel[6:18])
            torques = p_gains * (action * actions_scale + default_dof_pos - dof_pos) - d_gains * dof_vel
            torques = np.clip(torques, -torque_limits, torque_limits)
            data.ctrl[0:12] = torques
            # 执行仿真
            mujoco.mj_kinematics(m, d)
            mujoco.mj_step(model, data)
            time.sleep(0.003)

        ###################
        # 实时输出command命令
        ###################
        if not np.array_equal(command, prev_command):
            print(
                "Use arrow keys to move the robot.\n",
                # "'l': turn left\n",
                # "'r': turn left\n",
                "UP: move forward,\n",
                "DOWN: move backward,\n",
                "LEFT: move left,\n" "RIGHT: move right.\n",
            )
            # print(
            #     "x_velocity: {:.2f},\ny_velocity: {:.2f},\nangular_velocity: {:.2f}\n".format(
            #         command[0], command[1], command[2]
            #     )
            # )
            print(
                "x: {:.2f},\ny: {:.2f},\nangular_velocity: {:.2f}\n".format(
                    command[0], command[1], command[2]
                )
            )
            prev_command = command.copy()

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 2
            # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        viewer.sync()
        viewer.user_scn.ngeom = 0
        # 1. 绘制目标位置的蓝色球体
        sphere_radius = 0.05
        sphere_size = np.array([sphere_radius, 0, 0]) # 对于球体，只有第一个元素代表半径
        sphere_rgba = np.array([0.0, 0.0, 1.0, 1.0])         # 蓝色 (R, G, B, Alpha)
        target_pos_ball = target_pos + np.array([0.0, 0.0, 0.3]) # 球体中心位置
        # 获取当前可用的几何体ID，并将其用于新几何体
        geom_id = viewer.user_scn.ngeom
        # 球体不需要特定的旋转，所以使用单位矩阵
        identity_mat = np.eye(3).flatten() # 展平的3x3单位矩阵
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id], # 传入要初始化的 mjvGeom 对象
            mujoco.mjtGeom.mjGEOM_SPHERE,   # 几何体类型：球体
            sphere_size,
            target_pos_ball,                # 球体的中心位置
            identity_mat,
            sphere_rgba
        )
        viewer.user_scn.ngeom += 1 # 增加自定义几何体计数

        # # 2. 绘制从机器人根部到目标位置的绿色连线
        # line_width = 0.01 # 线的粗细
        # line_rgba = np.array([0.0, 1.0, 0.0, 1.0]) # 绿色

        # geom_id = viewer.user_scn.ngeom
        # # mjv_connector 是一个方便的函数，用于绘制连接两个点的线条 [5]
        # mujoco.mjv_initGeom(
        #     viewer.user_scn.geoms[geom_id], # 传入要初始化的 mjvGeom 对象
        #     mujoco.mjtGeom.mjGEOM_SPHERE,   # 几何体类型：球体
        #     sphere_size,
        #     target_pos,                # 球体的中心位置
        #     identity_mat,
        #     sphere_rgba
        # )
        # viewer.user_scn.ngeom += 1

        # # 3. 绘制红色箭头：表示机器人前进方向和速度大小
        # # 将机器人局部坐标系下的前进向量转换到世界坐标系
        # q_w = body_quat[0]
        # q_vec = body_quat[1:]
        # local_forward_vec = np.array([1.0, 0.0, 0.0])
        # temp_vec = np.cross(q_vec, local_forward_vec) + q_w * local_forward_vec
        # fw_world = local_forward_vec + 2 * np.cross(q_vec, temp_vec)

        # # 根据机器人线速度的大小调整箭头长度
        # velocity_magnitude = np.linalg.norm(body_lin_vel)
        # # 应用缩放因子并限制箭头长度，防止过小或过大
        # head_len = min(velocity_magnitude * velocity_scale_factor, max_arrow_length)
        # if head_len < min_arrow_length: # 确保即使速度很小也有一个可见的箭头
        #     head_len = min_arrow_length

        # arrow_start = body_pos
        # arrow_end = body_pos + fw_world * head_len
        # arrow_rgba = np.array([1.0, 0.0, 0.0, 1.0]) # 红色
        # arrow_line_width = 0.02 # 箭头主体的粗细

        # # 绘制箭头主杆
        # geom_id = viewer.user_scn.ngeom
        # mujoco.mjv_connector(
        #     viewer.user_scn,
        #     geom_id,
        #     mujoco.mjtGeom.mjGEOM_LINE,
        #     arrow_line_width,
        #     arrow_start,
        #     arrow_end,
        #     arrow_rgba
        # )
        # viewer.user_scn.ngeom += 1

        # # 为箭头主杆添加“加粗”效果（通过绘制两条平行的偏移线）
        # offset = 0.02 # 世界坐标系中的微小偏移量
        # # 计算一个在XY平面上垂直于前进方向的2D法线向量
        # normal_2d = np.array([-fw_world[1], fw_world])
        # if np.linalg.norm(normal_2d) > 1e-6:
        #     normal_2d = normal_2d / np.linalg.norm(normal_2d) * offset
        # else:
        #     normal_2d = np.array([offset, 0.0]) # 如果没有水平移动，提供一个默认偏移

        # # 绘制第一条偏移线 (+normal)
        # geom_id = viewer.user_scn.ngeom
        # mujoco.mjv_connector(
        #     viewer.user_scn,
        #     geom_id,
        #     mujoco.mjtGeom.mjGEOM_LINE,
        #     arrow_line_width,
        #     arrow_start + np.array([normal_2d, normal_2d[1], 0.0]),
        #     arrow_end + np.array([normal_2d, normal_2d[1], 0.0]),
        #     arrow_rgba
        # )
        # viewer.user_scn.ngeom += 1

        # # 绘制第二条偏移线 (-normal)
        # geom_id = viewer.user_scn.ngeom
        # mujoco.mjv_connector(
        #     viewer.user_scn,
        #     geom_id,
        #     mujoco.mjtGeom.mjGEOM_LINE,
        #     arrow_line_width,
        #     arrow_start - np.array([normal_2d, normal_2d[1], 0.0]),
        #     arrow_end - np.array([normal_2d, normal_2d[1], 0.0]),
        #     arrow_rgba
        # )
        # viewer.user_scn.ngeom += 1

        # # 绘制箭头尖端的两条“叉”
        # angle = math.radians(30) # 叉与主杆之间的角度
        # prong_length = head_len * 0.2 # 每条叉的长度

        # # 提取世界坐标系前进向量的2D (XY) 分量并归一化，用于2D旋转
        # fw_xy_normalized = fw_world[:2]
        # if np.linalg.norm(fw_xy_normalized) > 1e-6:
        #     fw_xy_normalized = fw_xy_normalized / np.linalg.norm(fw_xy_normalized)
        # else:
        #     fw_xy_normalized = np.array([1.0, 0.0]) # 如果没有水平移动，使用默认方向

        # # 旋转2D前进向量以获得叉的方向
        # # 左叉方向
        # rot_left_x = fw_xy_normalized * math.cos(angle) - fw_xy_normalized[1] * math.sin(angle)
        # rot_left_y = fw_xy_normalized * math.sin(angle) + fw_xy_normalized[1] * math.cos(angle)
        # left_prong_dir = np.array([rot_left_x, rot_left_y, 0.0])

        # # 右叉方向
        # rot_right_x = fw_xy_normalized * math.cos(-angle) - fw_xy_normalized[1] * math.sin(-angle)
        # rot_right_y = fw_xy_normalized * math.sin(-angle) + fw_xy_normalized[1] * math.cos(-angle)
        # right_prong_dir = np.array([rot_right_x, rot_right_y, 0.0])

        # # 绘制左叉
        # geom_id = viewer.user_scn.ngeom
        # mujoco.mjv_connector(
        #     viewer.user_scn,
        #     geom_id,
        #     mujoco.mjtGeom.mjGEOM_LINE,
        #     arrow_line_width,
        #     arrow_end, # 从箭头主杆的尖端开始
        #     arrow_end - left_prong_dir * prong_length, # 向后延伸
        #     arrow_rgba
        # )
        # viewer.user_scn.ngeom += 1

        # # 绘制右叉
        # geom_id = viewer.user_scn.ngeom
        # mujoco.mjv_connector(
        #     viewer.user_scn,
        #     geom_id,
        #     mujoco.mjtGeom.mjGEOM_LINE,
        #     arrow_line_width,
        #     arrow_end, # 从箭头主杆的尖端开始
        #     arrow_end - right_prong_dir * prong_length, # 向后延伸
        #     arrow_rgba
        # )
        # viewer.user_scn.ngeom += 1

        # # 同步查看器以显示所有新添加的自定义几何体。 [4]
        # viewer.sync()

        # # 控制模拟速度，使其接近实时
        # step_start = time.time()
        # time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)

# # 进入你的 Conda 环境 lib 目录
# cd ~/anaconda3/envs/rlgpu/lib

# # 备份原有 libstdc++
# mkdir backup && mv libstdc*.so* backup/

# # 复制系统版本并创建软链接
# cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 .
# ln -s libstdc++.so.6 libstdc++.so
# ln -s libstdc++.so.6 libstdc++.so.6.0.19
