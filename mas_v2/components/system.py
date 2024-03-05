from components.imports import *    
from components.controller import *
from components.parameters import *

def f_linear(agent,t,controller):
    theta = controller(agent,t)
    return np.array([v*np.cos(np.deg2rad(theta)),v*np.sin(np.deg2rad(theta)),0])
    
def f_unicycle(agent,t,controller):
    omega = controller(agent,t)
    theta = agent.theta
    return np.array([v*np.cos((theta)),v*np.sin((theta)),omega])
    