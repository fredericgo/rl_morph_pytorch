import numpy as np
from os import path
from gym import utils
from gym.envs.mujoco import mujoco_env


class Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        mujoco_env.MujocoEnv.__init__(self, self.xml, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.sim.data.qpos[2]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[2]
        reward_ctrl = - 0.5 * np.square(a).sum()
        up_reward = (xposafter - xposbefore)/self.dt
        reward_jump = up_reward**2

        survive_reward = 1.0
        reward = reward_ctrl + reward_jump + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.26 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_jump=reward_jump,
            reward_ctrl=-reward_ctrl)

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
        qpos = []
        qvel = []
        qpos.append(s[:5])
        qvel.append(s[5:11])
        i = 11
        for b in self.model.body_names[2:]:
            body_id = self.sim.model.body_name2id(b)
            jnt_adr = self.sim.model.body_jntadr[body_id]
            if jnt_adr < 0:
                i += 2
                q = []
                v = []
            else:
                q = s[i:i+1]
                v = s[(i+1):(i+2)]
                i += 2

            qpos.append(q)
            qvel.append(v)

        qpos = np.concatenate(qpos)
        qvel = np.concatenate(qvel)
  
        xy = self.init_qpos[:2]
        qpos = np.concatenate([xy, qpos])
        self.set_state(qpos, qvel)

    @property
    def joint_slice(self):
        ndof = self.sim.model.nq - 2
        return slice(5, ndof)

if __name__ == "__main__":
    from gym.wrappers import TimeLimit

    env = Ant3()
    env = TimeLimit(env, 2000)

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        print(obs.shape)
        print(reward)
        if done:
            env.reset()

    