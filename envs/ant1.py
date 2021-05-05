import numpy as np
from gym import utils
from . import mujoco_env

class Ant1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.xml = 'ant1.xml'
        mujoco_env.MujocoEnv.__init__(self, self.xml, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        # minimize rotational velocity
        vr = self.get_body_xvelr("torso")
        rot_cost = .5 * np.square(vr).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = (forward_reward - ctrl_cost 
                 - contact_cost - rot_cost + survive_reward)
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.26 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_to_observation(self, s):
        nq = self.model.nq
        nv = self.model.nv
        xy = self.init_qpos[:2]
        qpos = np.concatenate([xy, s[:(nq-2)]])
        qvel = s[(nq-2):(nq-2+nv)]
        self.set_state(qpos, qvel)


if __name__ == "__main__":

    env = Ant1()

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()
