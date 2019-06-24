import gym
import numpy as np
import highway_env

env = gym.make("highway-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000
EPSILON = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

#DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_SIZE = [20] * 5
#print(DISCRETE_OS_SIZE)
discrete_os_win_size = (env.observation_space.high[0:5] - env.observation_space.low[0:5])/DISCRETE_OS_SIZE
discrete_os_win_size = discrete_os_win_size[0:5]
#nn print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
#nn print("Q_Table shape {}".format(q_table.shape))
#print(q_table)


def get_descrete_state(state):
	discrete_state = (state - env.observation_space.low[0:5]) / (discrete_os_win_size+1)
	#nn print("Inside discrete : {}".format(discrete_state))
	return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False

	obs = env.reset()
	obs = obs[0:5]
	#nn print("observation : {}".format(obs))
	discrete_state = get_descrete_state(obs)

	#print(discrete_state)
	#print(np.argmax(q_table[discrete_state]))

	done = False
	while not done:
		#nn print("Discrete State : {}".format(discrete_state))
		action = np.argmax(q_table[discrete_state])
		new_state, reward, done, info = env.step(action)
		new_state = new_state[0:5]
		new_discrete_state = get_descrete_state(new_state)
		#nn print("New state : {}".format(new_state))
		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			#print("max_future_q : {}".format(max_future_q))
			current_q = q_table[discrete_state + (action, )]

			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action, )] = new_q

		'''
		elif new_state[0] >= env.goal_position:
			print("We made it on episode : {}".format(episode))
			q_table[discrete_state + (action, )] = 0
		'''

		discrete_state = new_discrete_state

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		EPSILON -= EPSILON_DECAY_VALUE

	if episode % 100 == 0:
		print("Episode completed {}".format(episode))

env.close()







