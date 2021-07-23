from gym.envs.registration import register
import gym 
import os

import env


env = gym.make("ant-walk-v0")

env.reset()
for _ in range(1000):
    obs, reward, done, _ = env.step(env.action_space.sample())
    print(reward)
    if done:
        env.reset()
