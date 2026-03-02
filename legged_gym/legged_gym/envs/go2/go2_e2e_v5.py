"""
V5: Agile Explorer — Lower action and joint penalties to allow
"messier" but potentially much faster running gaits.
Hypothesis: Our sim2sim penalties might be preventing the discovery
of a highly energetic bounding gait needed to catch far balls.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV5Cfg(GO2E2ECfg):
    class commands(GO2E2ECfg.commands):
        max_speed = 3.0                # maximum allowed speed

    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Relaxed gait
            dof_vel = -0.001           # very light
            action_rate = -0.01        # very light
            action_smoothness = -0.001 # very light
            dof_pos = -0.5             # very light
            collision = -0.5           # allow some shuffling if it helps
            feet_air_time = 0.0        # don't force a specific gait

            # Driving goals
            velo_dir = 15.0            # fast forward
            task = 15.0                # high task


class GO2E2EV5CfgPPO(GO2E2ECfgPPO):
    class algorithm(GO2E2ECfgPPO.algorithm):
        entropy_coef = 0.02            # maximum exploration

    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v5_agile_explorer'
        experiment_name = 'go2_e2e_v5'
