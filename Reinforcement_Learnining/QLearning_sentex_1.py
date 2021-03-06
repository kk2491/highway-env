import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000
EPSILON = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
print(DISCRETE_OS_SIZE)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)
#print(q_table)


def get_descrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False

	discrete_state = get_descrete_state(env.reset())

	#print(discrete_state)
	#print(np.argmax(q_table[discrete_state]))

	done = False
	while not done:

		if np.random.random() > EPSILON:
			action = np.argmax(q_table[discrete_state])	
		else:
			action = np.random.randint(0, env.action_space.n)
		
		new_state, reward, done, info = env.step(action)
		new_discrete_state = get_descrete_state(new_state)
		#print("New state : {}".format(new_state))
		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			#print("max_future_q : {}".format(max_future_q))
			current_q = q_table[discrete_state + (action, )]

			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action, )] = new_q
		
		elif new_state[0] >= env.goal_position:
			print("We made it on episode : {}".format(episode))
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		EPSILON -= EPSILON_DECAY_VALUE

env.close()







