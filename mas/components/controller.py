from components.imports import *
from components.parameters import * 

def lin_controller(agent,t):
    angle = 45
    
    return angle
    
def uni_oscillation(agent,t):
    return 20*np.cos(t)

def cy_persuit(agent,t):
    tget = agent.target
    delx = tget.x - agent.x
    dely = tget.y - agent.y
    # print(agent.name,"delx,dely","{:.2f}".format(delx),"{:.2f}".format(dely))
    vec1 = np.array([delx,dely])
    th = agent.theta
    vec2 = np.array([np.cos(th),np.sin(th)])
    
    if np.linalg.norm(vec1) == 0:
        return 0
    cs = np.dot(vec1,vec2)/np.linalg.norm(vec1)
    sn = np.cross(vec1,vec2)
    # print(agent.name,"norm,sin","{:.2f}".format(np.linalg.norm(vec1)),"{:.2f}".format(sn))
    
    diff = np.arccos(cs)
    k = 1
    if sn>0:
        return np.abs(k*diff)*(-1)
    else:
        return np.abs(k*diff)
    