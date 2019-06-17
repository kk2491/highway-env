EPSILON_MIN = 0.005
MAX_NUM_EPISODES = 1
STEPS_PER_EPISODE = 200
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.005 # Learning rate
GAMMA = 0.98 # Discount factor
NUM_DISCRETE_BINS = 30 # Number of bins to Discretize each observation dim

import gym
import highway_env
import numpy as np
import random
from gym import wrappers

class Q_Learner(object):

	def __init__(self, env):
		self.obs_shape = env.observation_space.shape 
		self.obs_high  = env.observation_space.high 
		self.obs_low   = env.observation_space.low 
		self.obs_bins  = NUM_DISCRETE_BINS
		self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
		self.action_space = env.action_space.n 

		self.Q 		   = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.obs_bins + 1self.action_space))  # (51 x 51 x 3)
		self.gamma	   = GAMMA
		self.alpha	   = ALPHA
		self.epsilon   = 1.0


	def discretize(self, obs):
		#print("Before discretizing : ")
		#print(obs)
		discretized_obs = ((obs - self.obs_low) / self.bin_width).astype(int)
		#print("After discretizing : ")
		#print(discretized_obs) 
		return tuple(discretized_obs)


	def get_action(self, obs):
		discretized_obs = self.discretize(obs)
		#print("Discretized Obs : {}".format(discretized_obs))

		# Epsilob greedy action selection
		if self.epsilon > EPSILON_MIN:
			self.epsilon -= EPSILON_DECAY
		if np.random.random() > self.epsilon:
			print("Action from Q : {}".format(np.argmax(self.Q[:,discretized_obs])))
			return np.argmax(self.Q[:,discretized_obs])
		else:
			return np.random.choice([a for a in range(self.action_space)])


	def learn(self, obs, action, reward, next_obs):

		discretized_obs = self.discretize(obs)
		discretized_next_obs = self.discretize(next_obs)
		
		print("Q shape : {}".format(self.Q[:,discretized_obs].shape))
		#print("Q : {}".format(self.Q[:,discretized_obs]))
		
		
		td_target = reward + self.gamma * np.max(self.Q[:,discretized_next_obs])
		td_error  = td_target - self.Q[:,discretized_obs][action]
		self.Q[:,discretized_obs][action] += self.alpha * td_error



def train(agent, env):

	best_reward = -float('inf')
	for episode in range(MAX_NUM_EPISODES):
		done = False
		env.render()
		obs = env.reset()
		
		total_reward = 0.0
		while not done:
			action = agent.get_action(obs)
			#action = env.action_space.sample()
			print(action)
			next_obs, reward, done, info = env.step(action)
			agent.learn(obs, action, reward, next_obs)
			obs = next_obs
			total_reward += reward

		if total_reward > best_reward:
			best_reward = total_reward

		print("Episode : {} || Reward : {} || Best_reward : {} || EPS : {}".format(episode, total_reward, best_reward, agent.epsilon))

	#env.close()
	return np.argmax(agent.Q, axis = 2)


def test(agent, env, policy):
	
	done = False
	obs = env.reset()
	total_reward = 0.0

	while not done:
		current_obs = agent.discretize(obs)
		action = policy[:, current_obs]
		print(action)
		next_obs, reward, done, info = env.step(action)
		obs = next_obs
		total_reward += reward

	return total_reward


if __name__ == "__main__":
	
	env = gym.make("highway-v0")
	agent = Q_Learner(env)
	
	print("Agent Q : {}".format(agent.Q))
	print("Observation Space : {}".format(agent.obs_shape))
	print("Observation Space High : {}".format(agent.obs_high))
	print("Observation space Low  : {}".format(agent.obs_low))
	print("Observation Bins       : {}".format(agent.obs_bins))
	print("Observation Bin Width  : {}".format(agent.bin_width))

	
	learned_policy = train(agent, env)
	
	print("Training Completed")

	gym_monitor_path = "./monitor_output"
	env = gym.wrappers.Monitor(env, gym_monitor_path, force = True)

	print("Testing Started")
	for _ in range(1000):
		test(agent, env, learned_policy)

	print("Testing Completed")
	
	env.close()
	
	
