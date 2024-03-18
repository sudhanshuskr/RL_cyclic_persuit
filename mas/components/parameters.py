import numpy as np



spawn_start = 0
spawn_end = 0

spawn_radius = 2
alpha = 1
dt = 1
v = 0.1
r0 = 5
alim = 2*np.pi
nx = 20
ny = 20
na = 10
xlim = nx/2
ylim = ny/2
N = 3
L = 3
d0 = np.cos((2*np.pi)/(2*N))*r0*2
cost_alpha = 0.5
EPSILON_TRAIN = 1
EPSILON_TEST = 1
learning_rate = 0.7
EPISODES = 100
GAMMA = 0.8
EPOCH = 80
TAU = 200
ESTIMATION_FREQ = 1
spawn_center = (-7,-7)
EPISODE_LENGTH = 350
step_ahead_w = 0

SAVE = 0
PLOT = 0