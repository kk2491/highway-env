import gym
import highway_env
import random

env = gym.make("highway-v0")

obs = env.reset()

done = False

while not done:
	action = 2
	next_state, reward, done, info = env.step(action)
	obs = next_state
	env.render()

env.close()
