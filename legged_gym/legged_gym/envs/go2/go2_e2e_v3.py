"""
V3: Higher Max Speed Command — The command range itself might be the blocker.
Hypothesis: If command max_speed is 1.5, the robot will never learn to run at 2.5m/s
even with infinite velo_dir. Here we increase the command boundary.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV3Cfg(GO2E2ECfg):
    class commands(GO2E2ECfg.commands):
        max_speed = 2.5                # Allow much faster movement!

    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            velo_dir = 15.0
            exploration = 4.0
            feet_air_time = 4.0        # more air time needed for higher speeds


class GO2E2EV3CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v3_high_speed_cmd'
        experiment_name = 'go2_e2e_v3'
