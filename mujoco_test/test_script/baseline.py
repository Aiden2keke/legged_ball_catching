#!/usr/bin/env python3

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
unitree_mujoco_test_dir = os.path.join(current_dir, "..")
sys.path.append(unitree_mujoco_test_dir)
actor_dir = os.path.join(unitree_mujoco_test_dir, "model/blind_locomotion/actor")
proprio_encoder_dir = os.path.join(unitree_mujoco_test_dir, "model/blind_locomotion/proprio_encoder")
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
import random
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
    "m": False,
    "space": False,  # 添加空格键用于扔球
    "backspace": False
}
# 最大和最小速度限制
max_speed = 0.5
min_speed = -0.5
# 阻尼系数，用于速度平滑下降
damping_factor = 0.2  # 调整该值以改变速度衰减快慢

# 按键按下回调
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
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["l"] = True
    elif key == keyboard.KeyCode.from_char("m"):
        key_state["m"] = True
    elif key == keyboard.Key.space:
        key_state["space"] = True
    elif key == keyboard.Key.backspace:
        key_state["backspace"] = True

# 按键释放回调
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
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["l"] = False
    elif key == keyboard.KeyCode.from_char("m"):
        key_state["m"] = False
    elif key == keyboard.Key.space:
        key_state["space"] = False
    elif key == keyboard.Key.backspace:
        key_state["backspace"] = False

def f(v, theta_rad, h_r=0.25):
    g = 9.81
    t_1 = v * math.sin(theta_rad) / g
    h_1 = g * t_1 ** 2 / 2
    # handle small numerical negatives safely
    delta_h = h_1 - h_r
    if delta_h < 0:
        # apex below reference height -> no real falling time, return NaNs
        return float('nan'), float('nan')
    t_2 = math.sqrt(2 * delta_h / g)
    t = t_1 + t_2
    x = -v * math.cos(theta_rad) * t
    return t, x

def get_command(body_pos, target_pos, body_quat, dt, max_speed=1.15, max_yaw_rate=0.5, kp=2.0):
    """
    纯 P 控制器：v = kp * (target_pos - body_pos)
    返回机体坐标系下的速度指令 (vx, vy, vz)
    """
    pos_diff = target_pos - body_pos

    distance = np.linalg.norm(pos_diff[:2])
    heading = np.arctan2(pos_diff[1], pos_diff[0])
    fw = quat_rotate_inverse(body_quat, np.array([1, 0, 0]))
    current_heading = np.arctan2(fw[1], fw[0])

    command = np.zeros(3)
    if distance > 0.05:
        # vx, vy
        v_world = kp * quat_rotate_inverse(body_quat, pos_diff)[:2]
        command[:2] = v_world[:2]

        # vx, yaw
        # command[0] = kp * distance
        # command[2] = kp * (heading - current_heading)

        command[2] = -np.clip(kp * (heading - current_heading), -max_yaw_rate, max_yaw_rate)
    else:
        command[:] = 0.0
    return command

##############
# 加载机器人模型
##############
model = mujoco.MjModel.from_xml_path(robot_data_dir + "/go2/scene_terrain.xml")
data = mujoco.MjData(model)

############################
# 加载 policy 和 encoder 模型
############################
device = torch.device("cuda")
actor = Actor(num_obs=77, num_actions=12, hidden_dims=[512, 256, 128])
# actor.load_state_dict(torch.load(actor_dir + "/actor_baseline.pth"))
actor.load_state_dict(torch.load(actor_dir + "/actor_oraclebl-9000.pth"))
actor = actor.to(device)
actor.eval()
proprio_encoder = MLPEncoder(input_dim=45*15)
# proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + "/proprio_baseline.pth"))
proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + "/proprio_oraclebl-9000.pth"))
proprio_encoder = proprio_encoder.to(device)
proprio_encoder.eval()

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
target_pos = np.zeros(3)
target_time = 0.0
pos_diff = np.zeros(3)
distance = 0.0
dt = 0.02  # 仿真步长（可根据实际设置）
# 扔球
ball_thrown = False
throw_ball = np.zeros(3)
throw_ball_x = 3.5
throw_ball_y_max = 0.01
ball_vel_max = 7.0
ball_vel_min = 6.99
ball_theta_max = 80.0
ball_theta_min = 65.0
base_height = 0.25 # 机器人躯干的大致高度
ball_pos = np.zeros(3)
ball_initial_vel = np.zeros(3)
ball_vel = np.zeros(3)
# 获取球的自由关节索引
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
# print("ball_joint_id:", ball_joint_id)
ball_qpos_idx = model.jnt_qposadr[ball_joint_id]  # 位置索引
ball_qvel_idx = model.jnt_dofadr[ball_joint_id]   # 速度索引

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

##############
# 关节状态初始化
##############
data.qpos[7:19] = default_dof_pos

#########################
# 监听键盘输入以发送 command
#########################
command = np.array([0.0, 0.0, 0.0])
prev_command = np.array([0.0, 0.0, 0.0])

# 监听键盘输入
command_scale_factor = 0.005
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
proprio_obs_history = torch.zeros((1, 45*15)).to(device)

###########
# 仿真主循环
###########
m = model
d = data

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
        t += dt
        # 检查空格键是否被按下
        if key_state["space"]:
            ball_thrown = True
            print("Throw the ball!")
            key_state["space"] = False
        if key_state["backspace"]:
            # 重置机器人位置到原点
            data.qpos[0:3] = np.array([0.0, 0.0, 0.37])  # 原点+合适高度
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数，朝向重置
            command = np.zeros(3)  # 重置command
            key_state["backspace"] = False
            print("Robot reset to origin, command cleared.")

        if ball_thrown:
            # randomize ball throw parameters
            v = random.uniform(ball_vel_min, ball_vel_max)
            theta_rad = random.uniform(ball_theta_min, ball_theta_max) * math.pi / 180.0
            throw_ball_y = random.uniform(-throw_ball_y_max, throw_ball_y_max)
            target_pos_y = throw_ball_y
            # 球的初始位置和速度
            throw_ball = np.array([throw_ball_y, throw_ball_x, 0.0])
            # print("throw_ball:", throw_ball)
            ball_pos = throw_ball

            vx = v * np.cos(theta_rad)  # 水平速度分量
            vz = v * np.sin(theta_rad)  # 垂直速度分量
            # 假设只在 x-z 平面发射
            ball_initial_vel = np.array([0.0,-vx, vz])
            # print("ball_initial_vel:", ball_initial_vel)
            ball_vel = ball_initial_vel
            data.qpos[ball_qpos_idx:ball_qpos_idx+3] = ball_pos  # 球的位置
            data.qvel[ball_qvel_idx:ball_qvel_idx+3] = ball_vel  # 球的速度
            # 计算目标位置
            target_time, target_pos_x =f(v, theta_rad, base_height)
            target_pos = throw_ball + np.array([0.0, target_pos_x, base_height])
            ball_thrown = False
                        # find agility extreme
            pos_diff = target_pos - body_pos
            distance = np.linalg.norm(pos_diff[:2])
            heading = quat_rotate_inverse(body_quat, pos_diff)
            orientation = np.arctan2(heading[1], heading[0])
            print("pos_diff:", pos_diff)
            print("orientation:", orientation)
            print("distance:", distance)
# #处理数据
# # 进入文件所在目录（可选）
# cd /home/yd/program/legged_ball_catching/mujoco_test/test_script

# # 生成表格
# awk '
# /orientation:/  {o=$2}
# /distance:/     {d=$2; ++n; print n "\t" o "\t" d}
# ' find_agility_extreme.txt > result.tsv

# # 查看结果
# cat result.tsv

        # deploy的command法
        command = get_command(body_pos, target_pos, body_quat, dt)


        ############
        # 更新模型输入
        ############
        current_time = time.time()

        # body_lin_vel = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array(data.qvel[0:3]))
        body_ang_vel = np.array(data.qvel[3:6])  # mujoco 中的角速度是在局部坐标系下的，所以不需要转换
        gravity_projection = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array([0, 0, -1.0]))
        command_input = command
        # command_input=np.array([1.0,0.0,0.0]) # 直接一直让vx=1
        joint_pos = np.array(data.qpos[7:19])
        joint_vel = np.array(data.qvel[6:18])

        proprio_observation = np.concatenate(
            [
                body_ang_vel * body_ang_vel_scale,
                gravity_projection,
                command_input * command_scale,
                (joint_pos - default_dof_pos) * joint_pos_scale,
                joint_vel * joint_vel_scale,
                last_action,
            ]
        )

        noise_strength = 0.02  # 噪声的强度，值越大，噪声越强
        noise = np.random.randn(*proprio_observation.shape) * noise_strength  # 生成与 proprio_observation 形状相同的噪声
        proprio_observation += noise  # 将噪声添加到 proprio_observation# 加入噪声

        proprio_observation = torch.from_numpy(proprio_observation).float().to(device).unsqueeze(0) # shape (1,45)
        proprio_obs_history = torch.cat((proprio_obs_history[:,45:], proprio_observation),dim=-1) # shape (1,675)

        #####################################
        # 将 observation 输入模型得到输出 action
        #####################################
        proprio_latent = proprio_encoder(proprio_obs_history)
        proprio_latent = torch.nn.functional.normalize(proprio_latent, p=2, dim=-1)
        actor_input = torch.cat((proprio_observation, proprio_latent), dim=-1)
        
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
            # print(
            #     "Use arrow keys to move the robot.\n",
            #     "'l': turn left\n",
            #     "'r': turn left\n",
            #     "UP: move forward,\n",
            #     "DOWN: move backward,\n",
            #     "LEFT: move left,\n" "RIGHT: move right.\n",
            # )
            # print(
            #     "x_velocity: {:.2f},\ny_velocity: {:.2f},\nangular_velocity: {:.2f}\n".format(
            #         command[0], command[1], command[2]
            #     )
            # )
            prev_command = command.copy()

        with viewer.lock():
            body_pos = np.array(data.qpos[0:3])
            desired_lookat = body_pos.copy()
            desired_lookat[2] += 0.2  # 把相机焦点抬高一点，避免贴地

    # # 直接设置（最直接，但可能会比较抖）
    #         viewer.cam.lookat = desired_lookat.tolist()

    # # 可同时调整距离 / 视角
    #         viewer.cam.distance = 5.0   # 拉远或拉近第三人称视角
    #         viewer.cam.azimuth = 50    # 水平旋转角度
    #         viewer.cam.elevation = -20  # 垂直视角
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 2
            # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        viewer.sync()
        viewer.user_scn.ngeom = 0    
        
        # 1. 绘制目标位置的蓝色球体
        sphere_radius = 0.07 / 2
        sphere_size = np.array([sphere_radius, 0, 0]) # 对于球体，只有第一个元素代表半径
        sphere_rgba = np.array([0.0, 0.0, 1.0, 1.0])         # 蓝色 (R, G, B, Alpha)
        # 获取当前可用的几何体ID，并将其用于新几何体
        geom_id = viewer.user_scn.ngeom
        # 球体不需要特定的旋转，所以使用单位矩阵
        identity_mat = np.eye(3).flatten() # 展平的3x3单位矩阵
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id], # 传入要初始化的 mjvGeom 对象
            mujoco.mjtGeom.mjGEOM_SPHERE,   # 几何体类型：球体
            sphere_size,
            target_pos,                # 球体的中心位置
            identity_mat,
            sphere_rgba
        )
        viewer.user_scn.ngeom += 1 # 增加自定义几何体计数
# # 进入你的 Conda 环境 lib 目录
# cd ~/anaconda3/envs/rlgpu/lib

# # 备份原有 libstdc++
# mkdir backup && mv libstdc*.so* backup/

# # 复制系统版本并创建软链接
# cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 .
# ln -s libstdc++.so.6 libstdc++.so
# ln -s libstdc++.so.6 libstdc++.so.6.0.19
