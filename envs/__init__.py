
from gym.wrappers import TimeLimit

from .ant import AntEnv


def load(env_name):
    if env_name == "ant":
        env = AntEnv()

    wrapped = TimeLimit(env, 1000)
    return wrapped