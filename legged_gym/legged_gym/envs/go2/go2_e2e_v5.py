"""
V5: Gait Regularizer
Extremely stiff gait penalties to force a very pretty and conservative walk, preventing any weird behavior even at the cost of some reach.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV5Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            dof_vel = -0.01
            action_rate = -0.08
            action_smoothness = -0.005
            feet_air_time = 3.0


class GO2E2EV5CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v5_gait_regularizer'
        experiment_name = 'go2_e2e_v5'
