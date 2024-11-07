#!/bin/bash
itraj=$1
noise=$2
echo $noise
echo $itraj
cd ~/qmon-sindy
. ~/qenv_bilkis/bin/activate
python3 mp_runs/sinin1_15.py --itraj $itraj --noise_level $noise
deactivate
