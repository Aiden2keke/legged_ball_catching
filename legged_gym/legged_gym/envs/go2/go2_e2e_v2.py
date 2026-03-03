"""
V2: The Perfect Stopper (完美刹车)
极端的刹车和固定姿势。目的是看看强制在目标点冻结姿态是否能提高抓球率。
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV2Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            velo_lim = -12.0           # 极其实严苛的限速
            stand_still_pos = -3.0     # 强制在目标点冻结姿态


class GO2E2EV2CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v2_the_perfect_stopper'
        experiment_name = 'go2_e2e_v2'
