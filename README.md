# Legged Ball Catching

This project is based on [legged_gym](https://github.com/leggedrobotics/legged_gym) and provides a framework for training, simulating, and deploying reinforcement learning policies for legged robots.

## Directory Structure
The project adopts a modular design, clearly separating training, simulation, and deployment functions for easier development and maintenance:
```plaintext
legged_ball_catching/
├── legged_gym/    # Reinforcement learning framework based on NVIDIA Isaac Gym
├── mujoco_test/   # MuJoCo simulation environment and test scripts
│ ├── data/        # Robot model and environment configuration files
│ ├── test_script/ # Simulation test scripts
│ └── model/       # Trained policy model
├── deployment/    # Real robot deployment code
├── rsl_rl/        # Reinforcement learning algorithm implementation
├── README.md      # Project documentation
├── lcm_daemon.sh  # LCM daemon startup script
└── run_policy.sh  # Policy execution script
```

---

## Environment Setup

This project involves three main stages: training, MuJoCo simulation, and deployment.  
**Please follow the recommended environment setup for each stage to ensure compatibility and reproducibility.**

---

### 1. Clone the Repository

```bash
git clone https://github.com/Aiden2keke/legged_ball_catching.git
cd legged_ball_catching
```

### 2. Training Environment

The training code is based on [legged_gym](https://github.com/leggedrobotics/legged_gym). **Please follow the installation part in legged_gym docs for details to set up the environment and dependencies.**

---

### 3. MuJoCo Simulation Environment

For MuJoCo-based simulation and visualization, install MuJoCo and its Python bindings in your environment:

```bash
pip install mujoco==3.2.2 mujoco-python-viewer
```

You may also need:
```bash
pip install pynput matplotlib
```

**Note:**  
- Use a Python environment compatible with MuJoCo (Python 3.8+ recommended).
- For more details, refer to [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/).

---

### 3. Deployment Environment

For deployment on real robots, please follow the environment setup and deployment instructions from  
[walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2?tab=readme-ov-file).

**Key steps:**
- Build and install the deployment code as described in the above repository.
- Ensure all hardware and software dependencies are met for your robot platform.

---

**Summary:**  
- **Training:** Follow [legged_gym](https://github.com/leggedrobotics/legged_gym?tab=readme-ov-file#installation) for environment and code base.
- **MuJoCo Simulation:** Install MuJoCo and related Python packages as above.
- **Deployment:** Follow [walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2?tab=readme-ov-file#requirements-1) for environment and deployment steps.

If you encounter issues, please check the documentation of each upstream project or open an issue in this repository.

## Training

### 1. Teacher-Student Training

Train a policy using the teacher-student framework: 

```python legged_gym/scripts/train.py --task=go2 --headless```

### 2. Student Reinforcing Training

Continue training or reinforce a student policy:

```python legged_gym/scripts/train.py  --task=go2 --headless --max_iterations=1500 --student_reinforcing --resume --experiment_name=rough_go2 --run_name={run_name} --checkpoint=7500```

### 3. Policy Evaluation (Play)

Evaluate a trained policy:

```python legged_gym/scripts/play.py --task=go2 --load_run={run_name} --checkpoint=9000```

---

## MuJoCo Simulation

To run simulation and visualization in MuJoCo, use the provided scripts in `mujoco_test/test_script/`.  
For example:

```bash
python mujoco_test/test_script/ball_catching.py
```

You can also run the simulation with the baseline policy as well.

---

## Deployment

Follow [walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2?tab=readme-ov-file#start-lcm) for the deployment steps.

First run lcm_position_go2 in the background:
```Bash
sudo ~/program/legged_ball_catching/deployment/go2_gym_deploy/build/lcm_position_go2 eth0
```
Run this before running deploy_policy.py
```Bash
cd deployment
export PYTHONPATH=$PYTHONPATH:`pwd`
cd go2_gym_deploy/scripts
python3 deploy_policy.py
```

---

## Additional Notes

- For custom tasks or environments, refer to the `legged_gym/envs` and `mujoco_test/test_script` directories.
- For troubleshooting or further customization, consult the official [legged_gym documentation](https://github.com/leggedrobotics/legged_gym) and [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/).
- If you encounter issues with dependencies, ensure your Python environment matches the recommended versions.

---

## Contact

For questions or contributions, please open an issue or pull request on the [GitHub repository](https://github.com/Aiden2keke/legged_ball_catching).