from skeleton_graph_vae.graph_dataset import GraphDataset
import envs
import gym
import utils
from graph_vae import wrappers
from graph_vae import graph_dataset
import numpy as np

ant = gym.make("envs:ant-v0")

ant = wrappers.ModularEnvWrapper(ant)
xml = ant.xml
g = utils.getGraphStructure(xml)

ds = GraphDataset("data/ant.memory", g, 27) 

s = ant.reset()

print(s.shape)