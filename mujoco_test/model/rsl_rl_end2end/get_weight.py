import os
import torch

run_name     = "10-1" # change to your log dir
# model_iters  = [5000, 6000, 7500, 10000, 13000, 15000, 20000, 25000, 30000]  # many checkpoints
model_iters  = [7500, 9000, 11500, 16500] # change to your iteration numbers
base_log_dir = "/home/yd/program/legged_ball_catching_end2end/legged_gym/logs/go2_e2e_v3" # change to your log dir

for model_iter in model_iters:
    # 1. Concatenate checkpoint paths based on variables
    ckpt_path = os.path.join(base_log_dir, run_name, f"model_{model_iter}.pt")
    if not os.path.isfile(ckpt_path):
        print(f"⚠️ no file: {ckpt_path}")
        continue
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    model_dict = ckpt["model_state_dict"]

    # 2. Extract proprioceptive_encoder
    prop_keys = [k for k in model_dict if k.startswith("proprioceptive_encoder.")]
    prop_dict = {
        k.replace("proprioceptive_encoder.", ""): model_dict[k]  # Remove prefix
        for k in prop_keys
    }

    # 3. Extract actor
    actor_dict = {k: v for k, v in model_dict.items() if k.startswith("actor.")}

    # 4. Save as .pth, the file name is also determined by the variable
    out_fname_prop = f"proprio_oracle_e2e{run_name}-{model_iter}.pth"
    torch.save(prop_dict, out_fname_prop)
    print(f"✅ save {out_fname_prop}")

    out_fname_actor = f"actor_oracle_e2e{run_name}-{model_iter}.pth"
    torch.save(actor_dict, out_fname_actor)
    print(f"✅ save {out_fname_actor}")
