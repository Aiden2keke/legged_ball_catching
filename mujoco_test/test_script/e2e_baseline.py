#!/usr/bin/env python3

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
unitree_mujoco_test_dir = os.path.join(current_dir, "..")
sys.path.append(unitree_mujoco_test_dir)
actor_dir = os.path.join(unitree_mujoco_test_dir, "model/rsl_rl_end2end/actor")
proprio_encoder_dir = os.path.join(unitree_mujoco_test_dir, "model/rsl_rl_end2end/proprio_encoder")
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
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v) * 2.0)

    return a - b + c

################################
# Initialize keyboard monitoring
################################
# Global variables, save keyboard status
key_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "l": False,
    "m": False,
    "space": False,  # Added spacebar for throwing balls
    "backspace": False
}

# Button press callback
def on_press(key):
    """Callback function when a button is pressed"""
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

# Key release callback
def on_release(key):
    """Callback function when the button is released"""
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

# Oblique projectile motion function
def landing_predictor(v, theta_rad, h_r=0.25):
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
    P controller: v = kp * (target_pos - body_pos)
    Returns the speed command in the body coordinate system (vx, vy, vz)
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

        command[2] = np.clip(kp * (heading - current_heading), -max_yaw_rate, max_yaw_rate)
    else:
        command[:] = 0.0
    return command

#########################
# Loading the robot model
#########################
model = mujoco.MjModel.from_xml_path(robot_data_dir + "/go2/scene_terrain.xml")
data = mujoco.MjData(model)

################################
# Load policy and encoder models
# NOTE: User needs to update this with their actual e2e run name/folder later.
################################
device = torch.device("cuda")
run_name = "10-1-9000"
actor = Actor(num_obs=48+32, num_actions=12, hidden_dims=[512, 256, 128])
actor.load_state_dict(torch.load(actor_dir + f"/actor_oracle_e2e{run_name}.pth"))
actor = actor.to(device)
actor.eval()

proprio_encoder = MLPEncoder(input_dim=48*5, hidden_dims=[512, 256, 128])
proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + f"/proprio_oracle_e2e{run_name}.pth"))
proprio_encoder = proprio_encoder.to(device)
proprio_encoder.eval()

######################################
# Initialize variables to store states
######################################
global command
body_pos = np.zeros(3)  # The robot's position in the world coordinate system
body_quat = np.zeros(4)  # The robot's orientation is in the world coordinate system
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
data.qpos[0:3] = np.array([0.0, 0.0, 0.37])  # 0.37 is the right height to prevent the robot from falling at the beginning
target_pos = np.zeros(3)
target_time = 0.0
pos_diff = np.zeros(3)
distance = 0.0
dt = 0.02  # Simulation step size (can be set according to actual conditions)
# throw ball parameters
ball_thrown = False
throw_ball = np.zeros(3)
throw_ball_x = 3.5
throw_ball_y_max = 0.01
ball_vel_max = 7.0
ball_vel_min = 6.99
ball_theta_max = 80.0
ball_theta_min = 65.0
base_height = 0.25 # Approximate height of the robot's base
ball_pos = np.zeros(3)
ball_initial_vel = np.zeros(3)
ball_vel = np.zeros(3)
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
ball_qpos_idx = model.jnt_qposadr[ball_joint_id]  # Position Index
ball_qvel_idx = model.jnt_dofadr[ball_joint_id]   # Velocity Index
zero_commands = False

########################################################
# Torque calculation related gains and observation scale
########################################################
p_gains = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])  # Position gains
d_gains = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Velocity gains
torque_limits = np.array([20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0])
actions_scale = 0.25
body_lin_vel_scale = 2.0
body_ang_vel_scale = 0.25
command_scale = np.array([2.0, 2.0, 0.25])
joint_vel_scale = 0.05
joint_pos_scale = 1.0

############################
# Joint state initialization
############################
data.qpos[7:19] = default_dof_pos

############################################
# Listen for keyboard input to send commands
############################################
command = np.array([0.0, 0.0, 0.0])
prev_command = np.array([0.0, 0.0, 0.0])
print(
    "Press 'space' to throw the ball.\n",
    "Press 'backspace' to reset the robot to the origin.\n",
)
# Listening for keyboard input
command_scale_factor = 0.005
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
proprio_obs_history = torch.zeros((1, 48*5)).to(device)

######################
# Simulation main loop
######################
m = model
d = data

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = [-7., 4.5, 3.]  # Example: move camera to (0, 0, 0.5)
    viewer.cam.azimuth = -40 # Example: rotate camera by 180 degree
    viewer.cam.elevation = -30  # Example: tilt camera by 30 degree
    start = time.time()
    last_update_time = start
    t = 0.0    
    while viewer.is_running() and time.time() - start < 5000:
        # Get the current robot position and orientation
        body_pos = np.array(data.qpos[0:3])
        body_quat = np.array(data.qpos[3:7])
        t += dt
        # Check if the spacebar is pressed
        if key_state["space"]:
            ball_thrown = True
            print("Throw the ball!")
            key_state["space"] = False
            zero_commands = False
        if key_state["backspace"]:
            # Reset the robot position to the origin
            zero_commands = True
            key_state["backspace"] = False
            print("Robot reset to origin, command cleared.")

        if ball_thrown:
            # randomize ball throw parameters
            v = random.uniform(ball_vel_min, ball_vel_max)
            theta_rad = random.uniform(ball_theta_min, ball_theta_max) * math.pi / 180.0
            throw_ball_y = random.uniform(-throw_ball_y_max, throw_ball_y_max)
            target_pos_y = throw_ball_y
            # The ball's initial position and velocity
            throw_ball = np.array([throw_ball_y, throw_ball_x, 0.0])
            ball_pos = throw_ball
            vx = v * np.cos(theta_rad) 
            vz = v * np.sin(theta_rad) 
            # Assume that emission is only in the x-z plane 
            ball_initial_vel = np.array([0.0,-vx, vz])
            ball_vel = ball_initial_vel
            data.qpos[ball_qpos_idx:ball_qpos_idx+3] = ball_pos
            data.qvel[ball_qvel_idx:ball_qvel_idx+3] = ball_vel
            # Calculate target position
            target_time, target_pos_x =landing_predictor(v, theta_rad, base_height)
            target_pos = throw_ball + np.array([0.0, target_pos_x, base_height])
            ball_throw_time = t
            ball_thrown = False
            # find agility extreme
            pos_diff = target_pos - body_pos
            distance = np.linalg.norm(pos_diff[:2])
            heading = quat_rotate_inverse(body_quat, pos_diff)
            orientation = np.arctan2(heading[1], heading[0])
            print("pos_diff:", pos_diff)
            print("orientation:", orientation)
            print("distance:", distance)

        if zero_commands:
            target_pos = body_pos.copy()
            body_quat = np.array([0.0, 0.0, 0.0, 0.0])
        command = get_command(body_pos, target_pos, body_quat, dt)


        ####################
        # Update model input
        ####################
        current_time = time.time()

        body_ang_vel = np.array(data.qvel[3:6])  # The angular velocity in mujoco is in the local coordinate system, so no conversion is required
        gravity_projection = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array([0, 0, -1.0]))
        
        # Calculate ball state in body frame
        ball_pos_world = np.array(data.qpos[ball_qpos_idx:ball_qpos_idx+3])
        ball_vel_world = np.array(data.qvel[ball_qvel_idx:ball_qvel_idx+3])
        
        # Check if ball is in flight
        try:
            ball_throw_time
        except NameError:
            ball_throw_time = -1000.0

        if ball_throw_time <= t <= ball_throw_time + target_time:
            # Transform ball pos to body frame
            ball_pos_body_true = quat_rotate_inverse(body_quat, ball_pos_world - body_pos)
            # Transform ball vel to body frame
            # (This is a simplified approx ignoring w x r, but matches env behavior of just rotating velocity vector)
            ball_vel_body_true = quat_rotate_inverse(body_quat, ball_vel_world)
            
            # Add visual noise to ball (simulating YOLO)
            ball_pos_noise = np.random.randn(3) * 0.05
            ball_vel_noise = np.random.randn(3) * 0.3
            
            ball_pos_body = ball_pos_body_true + ball_pos_noise
            ball_vel_body = ball_vel_body_true + ball_vel_noise
        else:
            # Hide the ball when not active so the robot doesn't react to it resting on ground
            ball_pos_body = np.zeros(3)
            ball_vel_body = np.zeros(3)

        joint_pos = np.array(data.qpos[7:19])
        joint_vel = np.array(data.qvel[6:18])

        proprio_observation = np.concatenate(
            [
                body_ang_vel * body_ang_vel_scale,
                gravity_projection,
                ball_pos_body,
                ball_vel_body,
                (joint_pos - default_dof_pos) * joint_pos_scale,
                joint_vel * joint_vel_scale,
                last_action,
            ]
        )

        noise_strength = 0.02  # General noise for proprio
        # We don't add noise to the ball again since it has its own noise profile above
        noise = np.random.randn(*proprio_observation.shape) * noise_strength
        noise[6:12] = 0.0 # Clear out general noise for the ball dims
        proprio_observation += noise

        proprio_observation = torch.from_numpy(proprio_observation).float().to(device).unsqueeze(0) # shape (1,48)
        proprio_obs_history = torch.cat((proprio_obs_history[:,48:], proprio_observation),dim=-1) # shape (1,240)

        #######################################################
        # Input observation into the model to get output action
        #######################################################
        proprio_latent = proprio_encoder(proprio_obs_history)
        proprio_latent = torch.nn.functional.normalize(proprio_latent, p=2, dim=-1)
        actor_input = torch.cat((proprio_observation, proprio_latent), dim=-1)

        action = actor(actor_input)
        action = action.detach().cpu().numpy()
        action = np.clip(action, -6.0, 6.0)
        last_action[:] = action[:]

        ########################################
        # Map action to torque as control output
        ########################################
        decimation = 4
        for i in range(decimation):
            # compute torque
            dof_pos = np.array(data.qpos[7:19])
            dof_vel = np.array(data.qvel[6:18])
            torques = p_gains * (action * actions_scale + default_dof_pos - dof_pos) - d_gains * dof_vel
            torques = np.clip(torques, -torque_limits, torque_limits)
            data.ctrl[0:12] = torques
            # Execute simulation
            mujoco.mj_kinematics(m, d)
            mujoco.mj_step(model, data)
            time.sleep(0.003)

        #############################
        # Output command in real time
        #############################
        if not np.array_equal(command, prev_command):
            prev_command = command.copy()


        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 2
        viewer.sync()
        viewer.user_scn.ngeom = 0    
        
        # Draw a blue sphere at the target location
        sphere_radius = 0.07 / 2
        sphere_size = np.array([sphere_radius, 0, 0])
        sphere_rgba = np.array([0.0, 0.0, 1.0, 1.0])
        geom_id = viewer.user_scn.ngeom
        identity_mat = np.eye(3).flatten()
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            sphere_size,
            target_pos,
            identity_mat,
            sphere_rgba
        )
        viewer.user_scn.ngeom += 1
