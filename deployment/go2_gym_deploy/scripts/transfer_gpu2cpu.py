import torch
import pickle
import glob

label = "gait-conditioned-agility/pretrain-go2/train"
dirs = glob.glob(f"../../runs/{label}/*")
logdir = sorted(dirs)[0]

# transfer
with open(logdir+"/parameters.pkl", 'rb') as file:
    pkl_cfg = pickle.load(file)
    # We check whether each value v in the dictionary is a PyTorch tensor via torch.is_tensor(v) .
    # If it is, we apply the .cpu() method to transfer it to the CPU; if not, we keep the original value.
    # This way, only actual tensors are transferred, and other types of values ​​(such as integers, strings, etc.) remain unchanged.
    pkl_cfg_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in pkl_cfg.items()}
    print("Transfer Succeed ! !")

# save transferred .pkl file
with open(logdir+"/parameters_cpu.pkl", 'wb') as file:
    pickle.dump(pkl_cfg_cpu, file)
    print("Transferred Pickle File has been saved as parameters_cpu.pkl")