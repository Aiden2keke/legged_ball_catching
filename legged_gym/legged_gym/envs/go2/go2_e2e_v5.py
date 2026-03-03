"""
V5: The Gait Regularizer (仪态大师)
最严苛的步态惩罚，确保绝对不会出现任何奇怪的爬行姿势。
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV5Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            dof_vel = -0.01
            action_rate = -0.08
            feet_air_time = 4.0


class GO2E2EV5CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v5_the_gait_regularizer'
        experiment_name = 'go2_e2e_v5'
