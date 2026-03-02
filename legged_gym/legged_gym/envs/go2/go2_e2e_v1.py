"""
V1: Fast Catch
Push speed but rely on the new base's stable gait and anti-overshoot.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV1Cfg(GO2E2ECfg):
    class commands(GO2E2ECfg.commands):
        max_speed = 2.5                # Allow much faster movement!

    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            # Speed constraints
            velo_dir = 15.0            # High base speed incentive
            feet_air_time = 4.0        # More air time needed for higher speeds


class GO2E2EV1CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v1_fast_catch'
        experiment_name = 'go2_e2e_v1'
