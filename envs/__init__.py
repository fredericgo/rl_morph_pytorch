
from gym.wrappers import TimeLimit

from .ant import AntEnv
from .ant1 import Ant1
from .ant2 import Ant2
from .ant3 import Ant3
from .ant_jump import AntJump
from .ant_turn import AntTurn
from .ant3_jump import Ant3Jump
from .ant3_turn import Ant3Turn
from .ant2_jump import Ant2Jump


def load(env_name):
    if env_name == "ant":
        env = AntEnv()
    elif env_name == "ant-jump":
        env = AntJump()
    elif env_name == "ant-turn":
        env = AntTurn()
    elif env_name == "ant1":
        env = Ant1()
    elif env_name == "ant2":
        env = Ant2()
    elif env_name == "ant2-jump":
        env = Ant2Jump()
    elif env_name == "ant3":
        env = Ant3()
    elif env_name == "ant3-jump":
        env = Ant3Jump()
    elif env_name == "ant3-turn":
        env = Ant3Turn()
    else:
        raise ValueError

    wrapped = TimeLimit(env, 1000)
    return wrapped