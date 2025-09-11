import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
unitree_mujoco_test_dir = os.path.join(current_dir, "..")
sys.path.append(unitree_mujoco_test_dir)
actor_dir = os.path.join(unitree_mujoco_test_dir, "model/rsl_rl_teacher_student/actor")
proprio_encoder_dir = os.path.join(unitree_mujoco_test_dir, "model/rsl_rl_teacher_student/proprio_encoder")
robot_data_dir = os.path.join(unitree_mujoco_test_dir, "data")

import time
import random
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

# 逆四元数旋转函数
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
    "space": False  # 添加空格键用于扔球
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

# 预测球在指定高度的落点和时间
def predict_ball_landing_at_height(ball_pos, ball_vel, base_height, gravity=np.array([0, 0, -9.81])):
    """
    预测球在 z=base_height 时的落点和时间。
    """
    z0 = ball_pos[2]
    vz0 = ball_vel[2]
    gz = gravity[2]
    # 解 0.5*gz*t^2 + vz0*t + (z0 - base_height) = 0
    a = 0.5 * gz
    b = vz0
    c = z0 - base_height
    discriminant = b**2 - 4*a*c
    if discriminant < 0 or a == 0:
        return np.zeros(3), 0.0
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2*a)
    t2 = (-b - sqrt_discriminant) / (2*a)
    # 选取正的最小根
    t_land = min([t for t in [t1, t2] if t > 0], default=0.0)
    if t_land <= 0:
        return np.zeros(3), 0.0
    # 计算落点
    land_pos = ball_pos[:2] + ball_vel[:2] * t_land
    return np.array([land_pos[0], land_pos[1], base_height]), t_land

# 添加球到模型
def add_ball_to_model(xml_path):
    # 直接读取XML文件内容
    try:
        with open(xml_path, 'r') as f:
            original_xml = f.read()
    except Exception as e:
        print(f"Error reading XML file: {e}")
        # 如果读取失败，返回原始模型创建
        return mujoco.MjModel.from_xml_path(xml_path)
    
    # 创建包含球的XML字符串片段
    ball_xml_snippet = """
    <body name=\"ball_body\" pos=\"2.0 0.0 1.0\">
        <joint type=\"free\" name=\"ball_joint\"/>
        <geom type=\"sphere\" size=\"0.05\" rgba=\"1.0 0.0 0.0 1.0\" mass=\"0.1\" contype=\"1\" conaffinity=\"1\"/>
    </body>
    """
    
    # 在worldbody标签中插入球的定义
    # 找到worldbody标签的结束位置
    worldbody_end_pos = original_xml.find("</worldbody>")
    if worldbody_end_pos == -1:
        # 如果找不到worldbody标签，直接在mujoco标签内添加
        mujoco_end_pos = original_xml.find("</mujoco>")
        if mujoco_end_pos == -1:
            # 如果找不到mujoco标签，返回原模型
            return mujoco.MjModel.from_xml_path(xml_path)
        modified_xml = original_xml[:mujoco_end_pos] + ball_xml_snippet + original_xml[mujoco_end_pos:]
    else:
        # 在worldbody标签内添加球
        modified_xml = original_xml[:worldbody_end_pos] + ball_xml_snippet + original_xml[worldbody_end_pos:]
    
    # 从修改后的XML创建新模型
    try:
        new_model = mujoco.MjModel.from_xml_string(modified_xml)
        return new_model
    except Exception as e:
        print(f"Error creating model with ball: {e}")
        # 如果创建失败，返回原始模型
        return mujoco.MjModel.from_xml_path(xml_path)

# 扔球函数（接受初始位置和速度作为参数）
def throw_ball(model, data, ball_pos, ball_vel):
    """
    设置球的位置和速度。
    """
    # 查找球的身体ID
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball_body")
    if ball_body_id < 0:
        return False
    
    # 设置球的位置和速度
    # qpos中，7-18是机器人，19-25是球。
    data.qpos[19:26] = np.concatenate([ball_pos, [1.0, 0.0, 0.0, 0.0]])  # 位置+四元数
    data.qvel[18:24] = ball_vel  # 线速度+角速度
    
    return True


##############
# 加载机器人模型
##############
# 加载机器人模型
model = mujoco.MjModel.from_xml_path(robot_data_dir + "/go2/scene_terrain.xml")
# 添加球到模型
model = add_ball_to_model(robot_data_dir + "/go2/scene_terrain.xml")
data = mujoco.MjData(model)

############################
# 加载 policy 和 encoder 模型
############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = "9-9000"
actor = Actor(num_obs=46+32, num_actions=12, hidden_dims=[512, 256, 128])  # 增加3个维度给落点，1个维度给时间
# actor.load_state_dict(torch.load(actor_dir + "/actor_oracle41-1-10000.pth"))
actor.load_state_dict(torch.load(actor_dir + f"/actor_oracle{run_name}.pth"))
actor = actor.to(device)
actor.eval()
proprio_encoder = MLPEncoder(input_dim=46*5)  # 注意：如果需要将球的预测信息也加入历史，这里需要相应调整
# proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + "/proprio_encoder_0910_cosine_reinforce.pth"))
proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + f"/proprio_oracle{run_name}.pth"))
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
# 球相关变量
ball_pos = np.zeros(3)
ball_vel = np.zeros(3)
ball_land_pos = np.zeros(3)
ball_land_time = 0.0
ball_thrown = False

# 默认关节角度
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
target_time = 0.0
dt = 0.02  # 仿真步长（可根据实际设置）
base_height = 0.25 # 机器人躯干的大致高度

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

# 关键参数表
# 为了增强报告的实用性，我们将关键参数整理成表格形式
print("表3: 关键仿真参数")
print("| 参数名称 | 值 | 描述 |")
print("|---|---|---|")
print(f"| p_gains | {p_gains.tolist()} | PD控制器位置项增益，控制关节 stiffness |")
print(f"| d_gains | {d_gains.tolist()} | PD控制器速度项增益，控制关节 damping |")
print(f"| torque_limits | {torque_limits.tolist()} | 关节力矩输出的物理限制 |")
print(f"| actions_scale | {actions_scale} | 策略动作的缩放因子 |")
print(f"| body_lin_vel_scale | {body_lin_vel_scale} | 躯干线速度观测的缩放因子 |")
print(f"| body_ang_vel_scale | {body_ang_vel_scale} | 躯干角速度观测的缩放因子 |")
print(f"| command_scale | {command_scale.tolist()} | 机器人命令（速度）的缩放因子 |")


# timer_active = False

##############
# 关节状态初始化
##############
data.qpos[7:19] = default_dof_pos

# 球状态初始化（如果模型中有球）
ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball_body")
if ball_body_id >= 0:
    # 隐藏球的初始位置
    data.qpos[7 + 12:7 + 12 + 3] = np.array([0.0, 0.0, 5.0])  # 初始位置远离机器人
    data.qvel[6 + 12:6 + 12 + 3] = np.zeros(3)  # 初始速度为0

#########################
# 监听键盘输入以发送 command
#########################
command = np.array([0.0, 0.0, 0.0])
prev_command = np.array([0.0, 0.0, 0.0])
print(
    "Use arrow keys to move the robot.",
    "UP: move forward,",
    "DOWN: move backward,",
    "LEFT: move left,",
    "RIGHT: move right.",
    "SPACE: throw a ball."
)
# 监听键盘输入
command_scale_factor = 0.005
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
proprio_obs_history = torch.zeros((1, 46*5)).to(device)

###########
# 仿真主循环
###########
m = model
d = data
velocity_scale_factor = 2.0  # 调整此值以使箭头长度在视觉上更明显
min_arrow_length = 0.05      # 即使速度为零，也确保箭头有一个最小可见长度
max_arrow_length = 1.0       # 限制箭头的最大长度，防止在高速时过长导致画面混乱
dist_history = torch.zeros(50).to(device)
ball_thrown_timer = 0.0 # 计时器，用于在球落地后禁止发球一段时间

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
        # arget_active = True
        t += dt
        # last_timer_active = timer_active
        # timer_active = False

        # 如果球已落地，开始计时
        if ball_thrown and np.linalg.norm(ball_vel) < 0.1 and ball_pos[2] < base_height:
             ball_thrown_timer += dt
        else:
             ball_thrown_timer = 0.0
             
        # 当球在地面上静止超过0.5秒后，允许再次发球
        if ball_thrown_timer > 0.5:
             ball_thrown = False
        
        # 检查空格键是否被按下，如果是则扔球
        if key_state["space"] and not ball_thrown:
            # 实现受约束的随机发球
            # 1. 随机选择一个在3m半径内的目标落点
            target_distance = random.uniform(0.5, 3.0)
            target_angle = random.uniform(0, 2 * np.pi)
            target_x = target_distance * np.cos(target_angle)
            target_y = target_distance * np.sin(target_angle)
            target_land_pos = np.array([target_x, target_y, base_height])
            
            # 2. 随机选择一个在1.5秒内的飞行时间
            flight_time = random.uniform(0.5, 1.5)
            
            # 3. 使用逆运动学计算初始速度
            gravity_accel = np.array([0, 0, -9.81])
            # 从机器人前方扔球
            throw_pos = body_pos + quat_rotate_inverse(body_quat, np.array([0.5, 0.0, 0.5]))  # 从机器人前方一定距离和高度扔球
            
            # 计算初始速度
            # v_x0 = (x_f - x_0) / t
            # v_y0 = (y_f - y_0) / t
            # v_z0 = (z_f - z_0 - 0.5*g*t^2) / t
            throw_vel_xyz = (target_land_pos - throw_pos - 0.5 * gravity_accel * flight_time**2) / flight_time
            throw_vel = np.concatenate([throw_vel_xyz, np.zeros(3)])  # 附加角速度为0
            # 4. 重新初始化球的位置并投掷
            throw_ball(model, data, throw_pos, throw_vel)
            ball_thrown = True
            print("Ball thrown with constrained trajectory!")
        
        # 检查球是否已经被扔出
        ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball_body")
        if ball_body_id >= 0 and ball_thrown:
            # 获取球的当前位置和速度
            ball_pos = np.array(data.qpos[19:22])
            ball_vel = np.array(data.qvel[18:21])
            
            # 预测球的落点和时间
            ball_land_pos, ball_land_time = predict_ball_landing_at_height(ball_pos, ball_vel, base_height)
            
            # 检查球是否已经落地
            # if ball_pos <= 0.05 and ball_vel <= 0.5:
            #     ball_thrown = False  # 重置扔球状态，允许再次扔球

        ############
        # 更新模型输入
        ############
        current_time = time.time()

        body_lin_vel = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array(data.qvel[0:3]))
        body_ang_vel = np.array(data.qvel[3:6])  # mujoco 中的角速度是在局部坐标系下的，所以不需要转换
        gravity_projection = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array([0, 0, -1.0]))
        joint_pos = np.array(data.qpos[7:19])
        joint_vel = np.array(data.qvel[6:18])

        # 准备球的预测信息（相对于机器人坐标系）
        if ball_thrown:
            # 将预测落点从世界坐标系转换到机器人坐标系
            pos_diff = ball_land_pos - body_pos
            relative_land_pos = quat_rotate_inverse(body_quat, pos_diff)
            command[:2] = relative_land_pos[:2]  # 使用相对于机器人的落点作为命令
            # 朝向误差（假设目标朝向为目标点方向）
            heading = np.arctan2(pos_diff[1], pos_diff[0])
            # 当前朝向
            fw = quat_rotate_inverse(body_quat, np.array([1, 0, 0]))
            current_heading = np.arctan2(fw[1], fw[0])
            # command[2] = heading - current_heading
            command[2] = np.clip(heading - current_heading, -0.3, 0.3)

            # 归一化处理预测时间
            normalized_land_time = min(ball_land_time / 20, 1.0)  # 假设最长预测时间为5秒
        else:
            command = np.zeros(3)
            normalized_land_time = 0.0

        # 扩展观测空间，添加球的预测信息
        proprio_observation = np.concatenate(
            [
                # body_lin_vel * body_lin_vel_scale,
                body_ang_vel * body_ang_vel_scale,
                gravity_projection,
                command,
                [normalized_land_time],
                (joint_pos - default_dof_pos) * joint_pos_scale,
                joint_vel * joint_vel_scale,
                last_action
            ]
        )

        proprio_observation = torch.from_numpy(proprio_observation).float().to(device).unsqueeze(0) # shape (1,46)
        proprio_obs_history = torch.cat((proprio_obs_history[:,46:], proprio_observation[:,:46]),dim=-1) # shape (1,46*5)

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
        # if not np.array_equal(command, prev_command):
        #     print(
        #         "x: {:.2f},
        # y: {:.2f},
        # angular_velocity: {:.2f}
        # ".format(
        #             command, command, command
        #         )
        #     )
        #     prev_command = command.copy()

        with viewer.lock():
            viewer.opt.flags = 2
        viewer.sync()
        viewer.user_scn.ngeom = 0
        
        # 绘制球的预测落点（如果有球被扔出）
        if ball_thrown and ball_land_time > 0:
            # 预测落点用黄色表示
            land_sphere_rgba = np.array([1.0, 1.0, 0.0, 0.8])  # 黄色，半透明
            geom_id = viewer.user_scn.ngeom
            sphere_radius = 0.05
            sphere_size = np.array([sphere_radius, 0, 0])
            identity_mat = np.eye(3).flatten()
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                sphere_size,
                ball_land_pos + np.array([0.0, 0.0, 0.05]),  # 稍微高于地面
                identity_mat,
                land_sphere_rgba
            )
            viewer.user_scn.ngeom += 1
            
            # 绘制从球到落点的虚线
            line_width = 0.01
            line_rgba = np.array([1.0, 1.0, 0.0, 0.5])  # 黄色，半透明
            
            if ball_body_id >= 0:
                ball_current_pos = np.array(data.qpos[19:22])
                geom_id = viewer.user_scn.ngeom
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_id],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.array([line_width, 0, 0]),
                    np.zeros(3),
                    identity_mat,
                    line_rgba
                )
                # 设置线段的两个端点
                viewer.user_scn.geoms[geom_id].pos = np.array([ball_current_pos, ball_current_pos, ball_current_pos,
                                                              ball_land_pos, ball_land_pos, ball_land_pos + 0.05])
                viewer.user_scn.ngeom += 1