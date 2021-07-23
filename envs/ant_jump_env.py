import numpy as np
from env.ant_like_env import AntLikeEnv


class Env(AntLikeEnv):
    def __init__(self, xml):
        super(Env, self).__init__(xml)

    def get_reward(self, a):
        xposbefore = self.sim.data.qpos[2]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[2]
        reward_ctrl = - 0.5 * np.square(a).sum()
        up_reward = (xposafter - xposbefore)/self.dt
        reward_jump = up_reward**2

        survive_reward = 1.0
        return reward_ctrl + reward_jump + survive_reward

     




