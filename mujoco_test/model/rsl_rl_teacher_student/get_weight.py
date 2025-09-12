import os
import torch

# 可配置的变量
run_name     = "12"
# model_iters  = [5000, 6000, 7500, 10000, 13000, 15000, 20000, 25000, 30000]  # 多个 checkpoint 步数
model_iters  = [9000, 11500, 16500]
base_log_dir = "/home/yd/program/legged_ball_catching/legged_gym/logs/rough_go2"

for model_iter in model_iters:
    # 1. 根据变量拼接 checkpoint 路径
    ckpt_path = os.path.join(base_log_dir, run_name, f"model_{model_iter}.pt")
    if not os.path.isfile(ckpt_path):
        print(f"⚠️ 文件不存在: {ckpt_path}")
        continue
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    model_dict = ckpt["model_state_dict"]

    # 2. 提取 proprioceptive_encoder
    prop_keys = [k for k in model_dict if k.startswith("proprioceptive_encoder.")]
    prop_dict = {
        k.replace("proprioceptive_encoder.", ""): model_dict[k]  # 移除前缀
        for k in prop_keys
    }

    # 3. 提取 actor
    # actor_keys = [k for k in model_dict if k.startswith("actor.")]
    # actor_dict = {
    #     k: model_dict[k]  # 保留原始键名称
    #     for k in actor_keys
    # }
    ##### 换一种方式提取 #####
    actor_dict = {k: v for k, v in model_dict.items() if k.startswith("actor.")}

    # 4. 保存为 .pth，文件名也由变量决定
    out_fname_prop = f"proprio_oracle{run_name}-{model_iter}.pth"
    torch.save(prop_dict, out_fname_prop)
    print(f"✅ 已保存 {out_fname_prop}")

    out_fname_actor = f"actor_oracle{run_name}-{model_iter}.pth"
    torch.save(actor_dict, out_fname_actor)
    print(f"✅ 已保存 {out_fname_actor}")
