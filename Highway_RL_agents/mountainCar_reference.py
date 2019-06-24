# Part 3 - Update to QLearning_sentex_1.py

import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 500
EPSILON = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
print(DISCRETE_OS_SIZE)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)
# print(q_table)

ep_rewards = []
aggr_ep_rewards = {"ep": [],
                   "avg": [],
                   "min": [],
                   "max": []}


def get_descrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_descrete_state(env.reset())

    # print(discrete_state)
    # print(np.argmax(q_table[discrete_state]))

    done = False
    while not done:

        if np.random.random() > EPSILON:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, info = env.step(action)

        episode_reward += reward

        new_discrete_state = get_descrete_state(new_state)
        # print("New state : {}".format(new_state))
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            # print("max_future_q : {}".format(max_future_q))
            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            # print("We made it on episode : {}".format(episode))
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        EPSILON -= EPSILON_DECAY_VALUE

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        np.save("q_tables_-q_table.npy", q_table)

        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards["max"].append(max(ep_rewards[-SHOW_EVERY:]))

        # print(f"Episode : {episode} Average : {average_reward} Minimum : {min(ep_rewards[-SHOW_EVERY])} Maximum : {max(ep_rewards[-SHOW_EVERY:])}")
        print("Episode : {} || Average : {} || Minimum : {} || Maximum : {}".format(episode, average_reward,
                                                                                    min(ep_rewards[-SHOW_EVERY:]),
                                                                                    max(ep_rewards[-SHOW_EVERY:])))

env.close()

import matplotlib.pyplot as plt

plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")
plt.legend(loc=4)
plt.show()
