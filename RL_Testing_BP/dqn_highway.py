import keras
import gym
import random
import numpy as np 
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import highway_env
import matplotlib.pyplot as plt

EPISODES = 1000

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen = 2000)
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim = self.state_size, activation = "relu"))
		model.add(Dense(24, activation = "relu"))
		model.add(Dense(self.action_size, activation = "linear"))
		model.compile(loss = "mse",
					  optimizer = Adam(lr = self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs = 1, verbose = 0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)


if __name__ == "__main__":
	render = False
	env = gym.make("highway-v0")
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n 
	agent = DQNAgent(state_size, action_size)
	#agent.load("DQN_10_OBS.h5")
	done = False
	batch_size = 32
	counter = 0
	episode_rewards = []
	for e in range(EPISODES):
		done = False
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		while not done:
			if render == True:
				env.render()
			action = agent.act(state)
			next_state, reward, done, info = env.step(action)
			
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			
			if len(agent.memory) > batch_size:
				agent.replay(batch_size)
			
			if e % 50 == 0:
				agent.save("DQN_HARD_REWARD_CHANGE.h5")
				render = True
			
			if counter >= 5:
            			render = False
            			counter = 0
		
		print("Episode : {}/{} || Reward : {} || Epsilon : {}".format(e, EPISODES, reward, agent.epsilon))
		
		episode_rewards.append(reward)
		counter += 1
	
	plt.plot(episode_rewards)
	plt.show()
	
			
			


