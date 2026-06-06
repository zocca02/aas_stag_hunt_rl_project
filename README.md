# Autonomous and Adaptive Systems: Stag Hunt
Project for the Autonomous and Adaptive Systems exam at the University of Bologna. Implementation of mutliple multiple agents interacting in Stag Hunt and Escalation social dilemma environments.

I implemented the agents mainly through reinforcement learning via DQN and I did some tests making LLMs playing as agents. For more details check the [report](report.pdf)\
The Gymnasium environment used is implemented in [Gym-Stag-Hunt](https://github.com/NullDefault/Gym-Stag-Hunt) and updated for Gymnasium by [Gymnasium-Stag-Hunt](https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt)

## Run the experiments
Implementation and tests were made with python 3.12

To run the local tests install all the required libraries, then download and install the environment

```
git clone https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt.git
cd Gymnasium-Stag-Hunt
pip install .
```

Then run the local tests with

```
.\run_tests.sh
```

Results will be saved in the savings folder

The LLMs experiments in  have been done on kaggle executing the [llm_stag_hunt.ipynb](llm_stag_hunt.ipynb) notebook\
If you want to visualize the results from the log run

```
python read_llm_logs.py
```
