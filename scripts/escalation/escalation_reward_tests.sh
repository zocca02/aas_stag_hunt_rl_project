#!/bin/bash

SHARED_PARAMS="-s -bf savings/escalation/ -E 1 -W 5 -H 5 -S 200 -e 1000 -A -Dse 1.0 -Dee 0.1 -Dd 0.6 -Derb 10000 -Dbs 256 -Dus 4 -g 0.9 -lr 5e-5 -a 0.6 -b 0.5 --beta-iters 50000 --ddqn-update-freq 400"

python3.12 dqn_escalation.py $SHARED_PARAMS -bp 0.1 -n "dqn_bp_-0.1"
python3.12 dqn_escalation.py $SHARED_PARAMS -bp 0.5 -n "dqn_bp_-0.5"
python3.12 dqn_escalation.py $SHARED_PARAMS -bp 1.0 -n "dqn_bp_-1.0"
python3.12 dqn_escalation.py $SHARED_PARAMS -bp 2.0 -n "dqn_bp_-2.0"

python3.12 dqn_escalation.py $SHARED_PARAMS -bp 0.1 -R -n "dqn_bp_-0.1_R"
python3.12 dqn_escalation.py $SHARED_PARAMS -bp 0.5 -R -n "dqn_bp_-0.5_R"
python3.12 dqn_escalation.py $SHARED_PARAMS -bp 1.0 -R -n "dqn_bp_-1.0_R"
python3.12 dqn_escalation.py $SHARED_PARAMS -bp 2.0 -R -n "dqn_bp_-2.0_R"