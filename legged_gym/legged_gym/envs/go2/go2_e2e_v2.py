"""
V2: Precision Stance (精准定点)
基于强化约束的新Base (go2_e2e_config.py)，进一步加强到达目标后的姿态固定和降速，彻底消灭过冲和乱跑。
期待：到达预判落点后，像钉子一样钉在地上，姿态异常稳定。
"""
from legged_gym.envs.go2.go2_e2e_config import GO2E2ECfg, GO2E2ECfgPPO


class GO2E2EV2Cfg(GO2E2ECfg):
    class rewards(GO2E2ECfg.rewards):
        class scales(GO2E2ECfg.rewards.scales):
            velo_lim = -15.0                   # 比Base更加严苛的限速
            stand_still_pos = -5.0             # 极度强制冻结姿态
            reach_pos_target_tight = 12.0      # 强化到达精准点的奖励
            tracking_position = 4.0            # 强化过程中的位置追踪
            stop_yaw_vel = -0.1                # 到达目标后严格禁止转向


class GO2E2EV2CfgPPO(GO2E2ECfgPPO):
    class runner(GO2E2ECfgPPO.runner):
        run_name = 'v2_precision_stance'
        experiment_name = 'go2_e2e_v2'
