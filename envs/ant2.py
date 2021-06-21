import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        mujoco_env.MujocoEnv.__init__(self, self.xml, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_ctrl = - 0.5 * np.square(a).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        survive_reward = 1.0
        vr = self.data.get_body_xvelr("torso")
        reward_rot = -.5 * np.square(vr).sum()
        reward = (reward_ctrl + reward_run + 
                  survive_reward + reward_rot)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.26 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_run=reward_run,
            reward_ctrl=-reward_ctrl,
            reward_rot=-reward_rot)

    def _get_obs(self):
        def _get_obs_per_limb(b):
            if b == 'torso':
                body_id = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[body_id]
                qpos_adr = self.sim.model.jnt_qposadr[jnt_adr] # assuming each body has only one joint
                angle = self.data.qpos[(qpos_adr+2):(qpos_adr+7)] # angle of 
                dof_adr = self.sim.model.jnt_dofadr[jnt_adr]
                qvel = self.data.qvel[dof_adr:(dof_adr+6)]
            else:
                body_id = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[body_id]
                qpos_adr = self.sim.model.jnt_qposadr[jnt_adr] # assuming each body has only one joint
                angle = self.data.qpos[qpos_adr] # angle of current joint, scalar
                dof_adr = self.sim.model.jnt_dofadr[jnt_adr]
                qvel = self.data.qvel[dof_adr]
                if jnt_adr < 0:
                    angle = [0.]
                    qvel = [0.]
                else:
                    angle = [angle]
                    qvel = [qvel]
            obs = np.concatenate([angle, qvel])
            return obs
        full_obs = np.concatenate([_get_obs_per_limb(b) for b in self.model.body_names[1:]])
        return full_obs.ravel()

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

    env = Ant2()

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        print(reward)
        if done:
            env.reset()
