from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
#import carla
import json
import numpy as np 
import time
import random
from gym.spaces import Box, Discrete, Tuple
import CarlaEnv

'''
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("/home/kishor/GWM/Carla_Precompiled/Carla_Compiled/Carla_Pre_Compiled/CARLA_0.8.2/CarlaUE4.sh"))
assert os.path.exists(SERVER_BINARY), "CARLA_SERVER environment variable is not set properly. Please check and retry"

try:
	from carla.client import CarlaClient
	from carla.sensor import Camera
	from carla.settings import CarlaSettings
	from carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, TURN_RIGHT, TURN_LEFT, LANE_FOLLOW
except ImportError:
	from .carla.client import CarlaClient
	from .carla.sensor import Camera
	from .carla.settings import CarlaSettings
	from .carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, TURN_LEFT, TURN_LEFT, LANE_FOLLOW	

# Parameter + Penanlty + Reward
HM_EPISODES = 10000

# Variables
epsilon			= 0.9
EPS_DECAY		= 0.9998
SHOW_EVERY  	= 100
LEARNING_RATE 	= 0.1
DISCOUNT		= 0.95

# Parameter Declaration
#COMMANDS_ENUM = {
#	REACH_GOAL : 
#}

# ACTION SPACE




# Custom observation space
def get_observation_space():

	return 1 

# Select the action from action space
def get_action(obs):
	return 1
	
	if np.random.random() > epsilon:
		action = np.argmax(q_table[obs])
	return action 
	

# 


if __name__ == "__main__":
	env = CarlaEnv()
	env = gym.make("Carla-v0")
	obs = env.reset()
	print("Initial : {}".format(obs))
	done = False

	while not done:
		obs = get_observation_space()
		action = get_action(obs)
		#next_state, reward, done, info = env.step(action)



'''
