import gym
import carla

# Parameter + Penanlty + Reward
HM_EPISODES = 10000

# Variables
epsilon			= 0.9
EPS_DECAY		= 0.9998
SHOW_EVERY  	= 100
LEARNING_RATE 	= 0.1
DISCOUNT		= 0.95

# Parameter Declaration
0 = 
1 = 
2 = 
3 = 
4 = 

# ACTION SPACE




# Custom observation space
def get_observation_space():

	return observation 

# Select the action from action space
def get_action(obs):
	if np.random.random() > epsilon:
		action = np.argmax(q_table[obs])
	return action 


# 


if __name__ == "__main__":

	env = gym.make("Carla-v0")
	obs = env.reset()
	print("Initial : {}".format(obs))
	done = False

	while not done:
		obs = get_observation_space()
		action = get_action(obs)
		next_state, reward, done, info = env.step(action)




