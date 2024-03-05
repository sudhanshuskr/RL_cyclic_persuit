import numpy as np


dt = 1
v = 0.1
r0 = 4
alim = 2*np.pi
nx = 20
ny = 20
na = 10
xlim = nx/2
ylim = ny/2
N = 5
d0 = np.cos((2*np.pi)/(2*N))*r0*2
cost_alpha = 0.9
EPSILON_TRAIN = 1
EPSILON_TEST = 1
learning_rate = 0.3
EPISODES = 100
GAMMA = 0.8
EPOCH = 150
TAU = 1000000
spawn_center = (-7,-7)
EPISODE_LENGTH = 8000000

SAVE = 0
PLOT = 1