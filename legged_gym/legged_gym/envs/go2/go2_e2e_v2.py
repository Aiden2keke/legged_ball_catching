"""
V2: Dense Tracking + Modest Speed — Balanced approach solving slowness
through dense rewards rather than raw speed incentives.
Hypothesis: Sometimes robots move slow because they are uncertain of the
path, not because they lack speed rewards.
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV2Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            velo_dir = 10.0            # slightly lower than base

            # Dense Tracking: VERY HIGH
            task = 8.0
            reach_landing_pos = 30.0   # Massive dense tracking
            track_ball_landing = 20.0  # Massive dense tracking
            tracking_position = 8.0
            tracking_yaw = 4.0


class GO2E2EV2CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v2_dense_tracker'
        experiment_name = 'go2_e2e_v2'
