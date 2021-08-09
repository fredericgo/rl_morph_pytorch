import numpy as np
from envs.ant_like_env import AntLikeEnv


class Env(AntLikeEnv):
    def __init__(self, xml):
        super(Env, self).__init__(xml)

    def get_reward(self, a):
        survive_reward = 1.0
        self.do_simulation(a, self.frame_skip)

        return survive_reward

   