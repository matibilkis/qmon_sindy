#!/bin/bash
itraj=$1
cd ~/qmon-sindy
. ~/qenv_bilkis/bin/activate
START=$(date +%s.%N)
python3 mp_runs/exp_dec.py --itraj $(($itraj))
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
done
deactivate
