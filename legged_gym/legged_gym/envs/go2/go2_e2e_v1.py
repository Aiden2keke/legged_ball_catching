"""
V1: The Sprinter (疾跑者)
在基础的稳定步态上，测试极限速度。跑得快，刹得更狠。
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV1Cfg(GO2E2ECfg):
    class commands(GO2E2ECfg.commands):
        max_speed = 3.0                # 超高速度上限

    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            velo_dir = 18.0            # 强烈推向前方
            velo_lim = -8.0            # 速度太快也要惩罚得更狠，要求强刹车
            feet_air_time = 4.0        # 更高的速度需要更多的腾空时间


class GO2E2EV1CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v1_the_sprinter'
        experiment_name = 'go2_e2e_v1'
