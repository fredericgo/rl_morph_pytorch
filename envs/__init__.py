
from gym.wrappers import TimeLimit

from .ant import AntEnv
from .ant1 import Ant1
from .ant2 import Ant2
from .ant3 import Ant3
from .ant_jump import AntJump

def load(env_name):
    if env_name == "ant":
        env = AntEnv()
    elif env_name == "ant-jump":
        env = AntJump()
    elif env_name == "ant1":
        env = Ant1()
    elif env_name == "ant2":
        env = Ant2()
    elif env_name == "ant3":
        env = Ant3()
    else:
        raise ValueError

    wrapped = TimeLimit(env, 1000)
    return wrapped