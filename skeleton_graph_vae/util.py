from __future__ import print_function
import os

import xmltodict

def getGraphStructure(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the graph structure as a parents list"""
    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch, self_idx)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    try:
        root = xml['mujoco']['worldbody']['body']
        assert not isinstance(root, list), 'worldbody can only contain one body (torso) for the current implementation, but found {}'.format(root)
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    # signal message flipping for flipped walker morphologies
    if 'walker' in os.path.basename(xml_file) and 'flipped' in os.path.basename(xml_file):
        parents[0] = -2
    return parents

def getGraphJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return all the joints defined as a list of tuples (body_name, joint_name1, ...) for each body"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    def preorder(b):
        if 'joint' in b:
            if isinstance(b['joint'], list) and b['@name'] != 'torso':
                raise Exception("The given xml file does not follow the standard MuJoCo format.")
            elif not isinstance(b['joint'], list):
                b['joint'] = [b['joint']]
            joints.append([b['@name']])
            for j in b['joint']:
                joints[-1].append(j['@name'])
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml['mujoco']['worldbody']['body']
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    return joints


def getMotorJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml['mujoco']['actuator']['motor']
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m['@joint'])
    return joints


class ModularEnvWrapper(gym.Wrapper):
    """Force env to return fixed shape obs when called .reset() and .step() and removes action's padding before execution"""
    """Also match the order of the actions returned by modular policy to the order of the environment actions"""
    def __init__(self, env, obs_max_len=None):
        super(ModularEnvWrapper, self).__init__(env)
        # if no max length specified for obs, use the current env's obs size
        if obs_max_len:
            self.obs_max_len = obs_max_len
        else:
            self.obs_max_len = self.env.observation_space.shape[0]
        self.action_len = self.env.action_space.shape[0]
        self.num_limbs = len(self.env.model.body_names[1:])
        self.limb_obs_size = self.env.observation_space.shape[0] // self.num_limbs
        self.max_action = float(self.env.action_space.high[0])
        self.xml = self.env.xml

        # match the order of modular policy actions to the order of environment actions
        self.motors = utils.getMotorJoints(self.env.xml)
        self.joints = utils.getGraphJoints(self.env.xml)
        self.action_order = [-1] * self.num_limbs
        for i in range(len(self.joints)):
            assert sum([j in self.motors for j in self.joints[i][1:]]) <= 1, 'Modular policy does not support two motors per body'
            for j in self.joints[i]:
                if j in self.motors:
                    self.action_order[i] = self.motors.index(j)
                    break

    def step(self, action):
        # clip the 0-padding before processing
        action = action[:self.num_limbs]
        # match the order of the environment actions
        env_action = [None for i in range(len(self.motors))]
        for i in range(len(action)):
            env_action[self.action_order[i]] = action[i]
        obs, reward, done, info = self.env.step(env_action)
        assert len(obs) <= self.obs_max_len, "env's obs has length {}, which exceeds initiated obs_max_len {}".format(len(obs), self.obs_max_len)
        obs = np.append(obs, np.zeros((self.obs_max_len - len(obs))))
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        assert len(obs) <= self.obs_max_len, "env's obs has length {}, which exceeds initiated obs_max_len {}".format(len(obs), self.obs_max_len)
        obs = np.append(obs, np.zeros((self.obs_max_len - len(obs))))
        return obs
