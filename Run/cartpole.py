"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
#from DQN.DDQN import Double_Dueling_DQN as DeepQNetwork#
from DQN.DQN_Setting import DQN_Setting
from DQN.DQN import DeepQNetwork
import matplotlib.pyplot as plt
from time import time
s=DQN_Setting()

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

s.n_action=env.action_space.n
s.n_feature=env.observation_space.shape[0]

RL = DeepQNetwork(env.action_space.n,env.observation_space.shape[0])
total_steps = 0

costs=[]
i_episodes=[]
cost=0
a=time()
for i_episode in range(500):
    observation = env.reset()
    ep_r = 0
    reward=0
    step=0
    while True:
        #env.render()
        #print(observation)
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > max(RL.batch_size,600):
            cost+=RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2),
                  'steps: ',step)
            i_episodes.append(step)
            costs.append(ep_r)
            cost=0
            step=0
            break
        observation = observation_

        total_steps += 1
        step+=1
print(time()-a)
plt.figure()
plt.plot(costs)
plt.figure()
plt.plot(i_episodes)
plt.show()