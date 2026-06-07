#!/bin/bash

SHARED_PARAMS="-s -bf savings/stag_hunt/ -E 1 -S 200 -e 2000 -A -sf -sr 5.0 -fr 1.0 -mp -1.5 -np 2 -Dse 1.0 -Dee 0.05 -Dd 0.5 -Derb 10000 -Dbs 64 -Dus 4 -g 0.9 -lr 1e-3 -a 0.6 -b 0.4 --beta-iters 100000"

python3.12 dqn_stag_hunt.py $SHARED_PARAMS -H 5 -W 5 -n "dqn_mp_-1.5_5x5"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -H 10 -W 10 -n "dqn_mp_-1.5_10x10"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -H 15 -W 15 -n "dqn_mp_-1.5_15x15"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -H 20 -W 20 -n "dqn_mp_-1.5_20x20"
