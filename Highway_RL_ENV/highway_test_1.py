import gym
import highway_env
env = gym.make("highway-v0")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODES = 500

for episode in range(MAX_NUM_EPISODES):
	
	obs = env.reset()
	done = False
	total_reward = 0.0
	step = 1
	
	while not done:
		env.render()
		action = env.action_space.sample()
		next_state, reward, done, info = env.step(action)
		total_reward += reward
		step += 1
		obs = next_state

	print("\n Episode : {} || Steps : {} || total_reward : {}".format(episode, step+1, total_reward))
		
env.close()
