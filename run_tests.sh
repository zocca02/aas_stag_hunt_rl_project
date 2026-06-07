#/bin/bash

./scripts/stag_hunt/stag_hunt_reward_tests.sh
./scripts/stag_hunt/stag_hunt_size_tests.sh

./scripts/escalation/escalation_reward_tests.sh

python3.12 create_plots.py