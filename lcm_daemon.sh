#!/bin/sh
proj_dir=$(realpath $(dirname $0))
sudo $proj_dir/deployment/go2_gym_deploy/build/lcm_position_go2 eth0
