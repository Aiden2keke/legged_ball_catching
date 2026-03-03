"""
V1: Fast Tracker (快速追踪)
基于强化约束的新Base (go2_e2e_config.py)，测试提升最高速度和追踪驱动力。
因为底层加上了严格的姿态约束 (-5.0) 和限速 (-12.0)，期待这版能在加大学习步长和速度驱动下依然平稳。
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV1Cfg(GO2E2ECfg):
    class commands(GO2E2ECfg.commands):
        max_speed = 2.5                # 适度提高速度上限，让他跑快点

    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            velo_dir = 15.0            # 增强向目标移动的驱动力
            feet_air_time = 4.0        # 更快速度通常需要更长的腾空


class GO2E2EV1CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v1_fast_tracker'
        experiment_name = 'go2_e2e_v1'
