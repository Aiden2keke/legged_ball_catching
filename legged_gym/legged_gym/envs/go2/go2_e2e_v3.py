"""
V3: Extreme Stiff Gait — Deepest penalties on joint movement and rate,
forcing the policy to find the most minimal-energy clean gait.
Hypothesis: Prevents the jitter often seen in sim2sim transfer.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO

class GO2E2EV3Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Extreme Sim2Sim constraints
            dof_vel = -0.01            # 2x base
            dof_acc = -2.0e-6          # 2x base
            action_rate = -0.1         # 2x base
            collision = -10.0          # 2x base
            action_smoothness = -0.01  # 2x base
            dof_pos = -6.0             # 1.5x base: ultimate rigidity

            # Speed
            velo_dir = 12.0
            exploration = 3.0
            velo_lim = -3.0

            # Task
            task = 10.0
            reach_landing_pos = 15.0
            track_ball_landing = 10.0
            feet_air_time = 8.0        # very high to force walking despite stiff joints

class GO2E2EV3CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v3_extreme_stiff'
        experiment_name = 'go2_e2e_v3'
