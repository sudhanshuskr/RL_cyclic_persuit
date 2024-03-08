import numpy as np

obs_x = -8
obs_y = -3
n_states = 7
FB_order = 7
n_weights = (FB_order + 1)**n_states

spawn_start = 0
spawn_end = 0

spawn_radius = 2
# alpha = 0.2
dt = 1
v = 0.1
r0 = 3
alim = 2*np.pi

xlim = 30
ylim = xlim
N = 4
L = 2
d0 = np.cos((2*np.pi)/(2*N))*r0*2
cost_alpha = 0.5
EPSILON_TRAIN = 1
EPSILON_TEST = 1
learning_rate = 0.2e-8
EPISODES = 40
GAMMA = 0.8
EPOCH = 40
TAU = 200
ESTIMATION_FREQ = 1
spawn_center = (-14,-14)
EPISODE_LENGTH = 400
step_ahead_w = 0

SAVE = 1
PLOT = 0

SARSA_n = 8