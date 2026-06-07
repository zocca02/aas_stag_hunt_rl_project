#!/bin/bash

SHARED_PARAMS="-s -bf savings/stag_hunt/ -E 1 -W 5 -H 5 -S 200 -e 2000 -A -sf -Dse 1.0 -Dee 0.05 -Dd 0.5 -Derb 10000 -Dbs 64 -Dus 4 -g 0.9 -lr 5e-4 -a 0.6 -b 0.4 --beta-iters 100000"

python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -0.5 -np 2 -n "dqn_mp_-0.5"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -1.5 -np 2 -n "dqn_mp_-1.5"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -2.5 -np 2 -n "dqn_mp_-2.5"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -3.5 -np 2 -n "dqn_mp_-3.5"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -5.0 -np 2 -n "dqn_mp_-5.0"

python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -0.5 -np 2 -R -n "dqn_mp_-0.5_R"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -1.5 -np 2 -R -n "dqn_mp_-1.5_R"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -2.5 -np 2 -R -n "dqn_mp_-2.5_R"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -3.5 -np 2 -R -n "dqn_mp_-3.5_R"
python3.12 dqn_stag_hunt.py $SHARED_PARAMS -sr 5.0 -fr 1.0 -mp -5.0 -np 2 -R -n "dqn_mp_-5.0_R"


