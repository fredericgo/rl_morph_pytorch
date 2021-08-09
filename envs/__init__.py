from gym.envs.registration import register
import gym 
import os
import importlib
from pathlib import Path

xml_dir = Path(__file__).resolve().parent / "xmls"
xmls = list(xml_dir.glob("*.xml"))
env_names = [x.name[:-4] for x in xmls]
env_names = sorted(env_names)

# shape-task
def register_env(env_name):
    shape, task = env_name.split("-")
    mod = importlib.import_module(f"envs.ant_{task}_env")
    fn = getattr(mod, "Env")
    params = {'xml': str(xml_dir / f"{shape}.xml")}
    register(id=(f"{env_name}-v0"),
                max_episode_steps=1000,
                entry_point=fn,
                kwargs=params)

env_names = [
    "ant4-walk",
    "ant4-jump",
    "ant3-walk",
    "ant3-jump",
    "ant_a-walk",
    "ant_b-walk",
    "ant_s1-walk",
    "ant5-walk",
    "ant6-walk",
    "ant7-walk",
    "ant8-walk",
    "ant3-rnd",
    "ant4-rnd",
    "ant5-rnd",
    "ant6-rnd"
]

for n in env_names:
    register_env(n)
