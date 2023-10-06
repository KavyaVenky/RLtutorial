import gymnasium as gym
import numpy as np
from datetime import datetime

"""
The maximum state in our environment is obtained by env.observation_space.high = [0.6, 0.07]
The lowest state in our envrionment is obtained by env.observation_space.low = [-1.2, -0.07]
Three sets of actions are possible in this environment. These are defined by the package. 0 - move left, 1 - stay in place, 2 - move right.

learning rate - [0, 1]. This value determines the speed at which the agent learns. It is important to have an optimum leanring rate to avoid unlearning.
discount rate - [0, 1]. Determines the percentage of importance that the agent should take when considering future rewards.
epsilon - a varying value that acts as a threshold value to determine if the agent should exploit its knowledge or explore random actions.

"""

# Calling the MountainCar environment from the package as it has predefined observation space and actions.
env = gym.make("MountainCar-v0", render_mode='human')
env.reset()

# defining important values
learning_rate = 0.1
discount = 0.95
episodes = 500 # number of runs for the agent to learn
epsilon = 1
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)


# observation space can be very huge if every floating point value between the highest and the lowest state has to be listed as a state. 
# So we turn them into a fixed no.of buckets of discrete values to keep it simple. The dimension of the size can vary between environments.
discrete_os_size = [20] * len(env.observation_space.high)

# range of values possible in the 20 chunks
window_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

def get_discrete_state(state):
	# converting continuous state values into one of the 20 chunks.
	discrete_state = (state - env.observation_space.low) / window_size
	return tuple(discrete_state.astype(int))


# initializing the Q table and assigning random values initially. [-2, 0] is the reward range.
q_table = np.random.uniform(low = -2, high = 0, size=(discrete_os_size + [env.action_space.n]))


print("Agent starts learning for the first time at", datetime.now())

for episode in range(episodes):
	discrete_state = get_discrete_state(env.reset()[0])
	done = False
	
	while not done:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state]) # exploit learnt knowledge.
		else:
			action = np.random.randint(0, env.action_space.n) # explore random actions.

		new_state, reward, done, flag, d = env.step(action)
		new_discrete_state = get_discrete_state(new_state)

		# update q table
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]

			new_q = (1 - learning_rate) * current_q +  learning_rate * (reward + discount * max_future_q)
			q_table[discrete_state+(action, )] = new_q

		elif new_state[0] >= env.goal_position:
			print("Agent reached the goal at", datetime.now())
			q_table[discrete_state + (action, )] = 0 # assign highest reward

		discrete_state = new_discrete_state

	if end_epsilon_decaying >= episode >= start_epsilon_decaying:
		epsilon -= epsilon_decay_value # gradually increase chances of exploiting learnt knowledge.

env.close()