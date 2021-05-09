from gym.envs.registration import register
import gym 
import os

from pathlib import Path

xml_dir = Path(__file__).resolve().parent / "xmls"
xmls = list(xml_dir.glob("*.xml"))
env_names = [x.name[:-4] for x in xmls]
env_names = sorted(env_names)


for env_name in env_names:  
    params = {'xml': str(xml_dir / f"{env_name}.xml")}
    register(id=(f"{env_name}-v0"),
                max_episode_steps=1000,
                entry_point=f"envs.{env_name}:Env",
                kwargs=params)