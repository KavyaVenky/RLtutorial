TERMINOLOGIES

1. environment(env) - The virtual space consisting of the hill, car, flag post.

2. observation_space - The dimensions of the observation space can change between environments. Here it is 2D namely position of the car, velocity of the car.

3. action_space - The set of unique actions that can be performed in the given environment.

4. state - Coordinates of the car's current location in the environment.

5. agent - In this case, the car.

6. Q table - A table with actions as columns and states as rows.


OVERVIEW

The carhill.py is a chunk of python code (not a model, just an intelligently evolving agent) that learns to drive uphill and reach the goal at the flag post. The code that you see is Q-learning, a type of reinforcement learning which maintains a q-table of actions and states and uses the same to make learnt decisions. In this way, the agent need not know anything about the environment.

SOFTWARE REQUIREMENTS

This code requires gymnasium. If you use the anaconda distribution of python. Run the following commands individually in the same order to fulfill these requirements. 

conda install -c conda-forge gymnasium
pip install gymnasium[classic-control]

P.S. As an alternate to the second command, you can run the following where you install pygame. However, I ran into some issues while with this and hence the above suggestion. However both these require installing gymnasium.

conda install -c cogsci pygame
