
from gym.wrappers import TimeLimit

from .ant import AntEnv
from .ant3 import Ant3

def load(env_name):
    if env_name == "ant":
        env = AntEnv()
    elif env_name == "ant3":
        env = Ant3()
    else:
        raise ValueError

    wrapped = TimeLimit(env, 1000)
    return wrapped