import gym
import highway_env
import random

env = gym.make("highway-v0")

done = False
obs = env.reset()
obs = obs[0:10]

while not done:
	print("Observation : {}".format(obs))
	action = 2
	next_state, reward, done, info = env.step(action)
	obs = next_state[0:10]
	env.render()
	
env.close()

