"""
V1: Conservative Sim2Sim — Slightly softer penalties than base,
allowing more exploration while still maintaining clean gait.
Hypothesis: Base config might be too strict and cause learning to stall.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV1Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Relaxed Sim2Sim constraints
            dof_vel = -0.003
            dof_acc = -5.0e-7
            action_rate = -0.02
            collision = -2.0
            action_smoothness = -0.002
            dof_pos = -2.0             # looser legs

            # Forward speed
            velo_dir = 10.0
            exploration = 2.0
            velo_lim = -2.0            # softer anti-overshoot

            # Task & Gait
            task = 8.0
            catch_bonus = 3.0
            reach_landing_pos = 12.0
            feet_air_time = 3.0

class GO2E2EV1CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v1_conservative_sim2sim'
        experiment_name = 'go2_e2e_v1'
