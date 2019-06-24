from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten 
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np 
import random
import time
import tensorflow as tf 
import os 
import cv2
from tqdm import tqdm
from PIL import Image 
import keras.backend.tensorflow_backend as backend

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "256x2"
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200
MEMORY_FRACTION = 0.20

EPISODES = 20000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATES_STATS_EVERY = 50
SHOW_PREVIEW = True

class Blob:
	def __init__(self, size):
		self.size = size
		self.x = np.random.randint(0, size)
		self.y = np.random.randint(0, size)

	def __str__(self):
		return "Blob ({}, {})".format(self.x, self.y)

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y

	def action(self, choice):

		if choice == 0:
			self.move(x = 1, y =1)
		elif choice == 1:
			self.move(x = -1, y = -1)
		elif choice == 2:
			self.move(x = -1, y = 1)
		elif choice == 3:
			self,move(x = 1, y = -1)

		elif choice == 4:
			self.move(x = 1, y = 0)
		elif choice == 5:	
			self.move(x = -1, y = 0)
		elif choice == 6:
			self.move(x = 0, y = 1)
		elif choice == 7:
			self.move(x = 0, y = -1)
		elif choice == 8:
			self.move(x = 0, y = 0)


	def move(self, x = False, y = False):
		if not x:
			self.x += np.random.randint(-1, 2)
		else:
			self.x += x

		if not y:
			self.y += np.random.randint(-1, 2)
		else:
			self.y += y

		if self.x < 0:
			self.x = 0
		elif self.x > self.size-1:
			self.x = self.size-1

		if self.y < 0:
			self.y = 0
		elif self.y > self.size-1:
			self.y = self.size-1


class BlobEnv:
	SIZE = 10
	RETURN_IMAGES = True 
	MOVE_PENALTY = 1
	ENEMY_PENALTY = 300
	FOOD_REWARD = 25
	OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
	ACTION_SPACE_SIZE = 9
	PLAYER_N = 1
	FOOD_N = 2
	ENEMY_N = 3

	d = {1 : (255, 175, 0),
		 2 : (0, 255, 0),
		 3 : (0, 0, 255)}

	def reset(self):
		self.player = Blob(self.SIZE)
		self.food = Blob(self.SIZE)
		while self.food == self.player:
			self.food = Blob(self.SIZE)
		self.enemy = Blob(self.SIZE)
		while self.enemy == self.player or self.enemy == self.food:
			self.enemy = Blob(self.SIZE)

		self.episode_step = 0

		if self.RETURN_IMAGES:
			observation = np.array(self.get_image())
		else:
			observation = (self.player - self.food) + (self.player - self.enemy)

		return observation

	def step(self, action):
		self.episode_step += 1
		self.player.action(action)

		if self.RETURN_IMAGES:
			new_observation = np.array(self.get_image())
		else:
			new_observation = (self.player - self.food) + (self.player - self.enemy)

		if self.player == self.enemy:
			reward = -self.ENEMY_PENALTY
		elif self.player == self.food:
			reward = self.FOOD_REWARD
		else:
			reward = -self.MOVE_PENALTY

		done = False
		if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 20:
			done = True

		return new_observation, reward, done


	def render(self):
		img = self.get_image()
		img = img.resize((300, 300))
		cv2.imshow("image", np.array(img))
		cv2.waitKey(1)

	def get_image(self):
		env = np.zeros((self.SIZE, self.SIZE, 3), dtype = np.uint8)
		env[self.food.x][self.food.y] = self.d[self.FOOD_N]
		env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
		env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
		img = Image.fromarray(env, "RGB")
		return img

env = BlobEnv()

ep_rewards = [-200]
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

if not os.path.isdir("models"):
	os.makedirs("models")



class ModifiedTensorBoard(TensorBoard):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.FileWriter(self.log_dir)

	def set_model(self, model):
		pass

	def on_epoch_end(self, epoch, logs = None):
		self.update_stats(**logs)

	def on_batch_end(self, batch, logs = None):
		pass

	def on_train_end(self, _):
		pass 

	def update_stats(self, **stats):
		self._write_logs(stats, self.step)


class DQNAgent:
	
	def __init__(self):
		
		# main model - train every step
		self.model = self.create_model()

		# target model - predict every step
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		self.tensorboard = ModifiedTensorBoard(log_dir = "logs/{}-{}".format(MODEL_NAME, int(time.time())))

		self.target_update_counter = 0

	def create_model(self):
		model = Sequential()
		model.add(Conv2D(256, (3, 3), input_shape = env.OBSERVATION_SPACE_VALUES))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(2, 2))
		model.add(Dropout(0.2))

		model.add(Conv2D(256, (3, 3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(2, 2))
		model.add(Dropout(0.2))

		model.add(Flatten())
		model.add(Dense((64)))

		model.add(Dense(env.ACTION_SPACE_SIZE, activation = "linear"))
		model.compile(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ["accuracy"])

		return model

	def update_replay_memory(self, transition):
		self.replay_memory.append(transition)

	def get_qs(self, terminal_state, step):
		return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

	def train(self, terminal_state, step):
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return

		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
 		
		current_states = np.array([transition[0] for transition in minibatch])/255
		current_qs_list = self.model.predict(current_states)

		new_current_states = np.array([transition[3] for transition in minibatch])/255
		future_qs_list = self.target_model.predict(new_current_states)

		X = []
		y = []

		for index, (current_states, action, reward, new_current_states, done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_future_q
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[index] = new_q

			X.append(current_state)
			y.append(current_qs)

		print("============Before breakpoint===========")
		print("X : {}".format(X))
		print("Y : {}".format(y))
		self.model.fit(np.array(X)/255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)
		print("============breakpoint===========")
		# to update the target_model
		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0


agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii = True, unit = "episodes"):
	agent.tensorboard.step = episode

	episode_reward = 0
	step = 1
	current_state = env.reset()

	done = False

	while not done:
		if np.random.random() > epsilon:
			action = np.argmax(agent.get_qs(current_state))
		else:
			action = np.random.randint(0, env.ACTION_SPACE_SIZE)

	new_state, reward, done = env.step(action)

	episode_reward += reward

	if SHOW_PREVIEW and not episode % AGGREGATES_STATS_EVERY:
		env.render()

	agent.update_replay_memory((current_state, action, reward, new_state, done))
	agent.train(done, step)

	current_state = new_state
	step += 1

	ep_rewards.append(episode_reward)
	if not episode % AGGREGATES_STATS_EVERY or episode == 1:
		average_reward = sum(ep_rewards[-AGGREGATES_STATS_EVERY:])/len(ep_rewards[-AGGREGATES_STATS_EVERY])
		min_reward = min(ep_rewards[-AGGREGATES_STATS_EVERY:])
		max_reward = max(ep_rewards[-AGGREGATES_STATS_EVERY:])
		agent.tensorboard.update_stats(reward_avg = average_reward, reward_min = min_reward, reward_max = max_reward)

		if min_reward >= MIN_REWARD:
			agent.model.save("models/{}_{}_max_{}_average_{}_min".format(max_reward, average_reward, min_reward))

	if epsilon > MIN_EPSILON:
		epsilon *= EPSILON_DECAY
		epsilon = max(MIN_EPSILON, epsilon)




































