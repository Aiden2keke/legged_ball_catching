"""
V3: Strong Brake
Extreme anti-overshoot focus.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV3Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            velo_lim = -10.0
            stand_still_pos = 5.0
            tracking_lin_vel = 2.0


class GO2E2EV3CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v3_strong_brake'
        experiment_name = 'go2_e2e_v3'
