#!/bin/bash
cd deployment
export PYTHONPATH=$PYTHONPATH:`pwd`
cd go2_gym_deploy/scripts
python3 deploy_policy.py
