from gym.envs.registration import register
import gym 
import os

import envs


env = gym.make("ant2-v0")

env.reset()
for _ in range(1000):
    obs, reward, done, _ = env.step(env.action_space.sample())
    print(reward)
    if done:
        env.reset()
