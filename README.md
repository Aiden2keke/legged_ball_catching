# Setup
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