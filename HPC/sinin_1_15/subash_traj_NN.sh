#!/bin/bash
itraj=$1
lr=$2
echo $lr
echo $itraj
cd ~/qmon-sindy
. ~/qenv_bilkis/bin/activate
python3 mp_runs/sinin1_15.py --itraj $itraj --lr $lr
deactivate
