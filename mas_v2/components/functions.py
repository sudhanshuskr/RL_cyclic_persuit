from components.imports import *
from components.rk4 import *
from components.u_classes import *



def isLastEpisode(episode):
    if episode == EPISODES - 1:
        return 1
    else:
        return 0
def nextState(team,dynamics,controller,t):
    ds = rk4_step(dynamics,team,t,controller)
    # if t< 0.05:
    #     print(ds)
    #     print("Angle in degrees : ", np.rad2deg(team[0].theta))
    i = 0
    for agent in team:
        agent.x = agent.x + ds[i][0]
        agent.y = agent.y + ds[i][1]
        agent.theta = agent.theta + ds[i][2]
        i = i+1
    # print("=========================================")
    


def get_posi_index(agent):
    x = agent.x
    y = agent.y
    return [int(np.floor(x-(-xlim))),int(np.floor(y-(-ylim)))]

def get_q_index(xa,ya,xt,yt,xc,yc):
    ret = []
    ret.append(int(np.floor(xa-(-xlim))))
    ret.append(int(np.floor(ya-(-ylim))))
    ret.append(int(np.floor(xt-(-xlim))))
    ret.append(int(np.floor(yt-(-ylim))))
    ret.append(int(np.floor(xc-(-xlim))))
    ret.append(int(np.floor(yc-(-ylim))))
    return ret
    

def  dist(a1,a2):
    dif = [a1.x - a2.x,a1.y - a2.y]
    dif = np.array(dif)
    return np.linalg.norm(dif)

def get_centroid(team):
    xc = 0
    yc = 0
    for ag in team:
        xc += ag.x
        yc += ag.y
    num_of_agents = len(team)
    xc = xc/num_of_agents
    yc = yc/num_of_agents
    return [xc,yc]


def J_i(agent,centroid,r0,d0,n):
    tg = agent.target
    di = dist(agent,tg)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy)
    Ji = 0
    Ji += 0.5*((di-d0)**2)*cost_alpha
    Ji += ((dc - r0)**2)*cost_alpha
    Ji += (1/n)*(cent_err**2)*(1 - cost_alpha)

    return Ji   

def J_i_PG(agent,centroid,r0,d0,n): #Proximal Gradient 
    tg = agent.target
    di = dist(agent,tg)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy)
    Ji = 0
    Ji += 0.5*((di-d0)**2)*cost_alpha
    Ji += ((dc - r0)**2)*cost_alpha
    Ji += (1/n)*(cent_err**2)*(1 - cost_alpha)

    return Ji   

def J_i_int(ax,ay,tx,ty,centroid,r0,d0,n): 
    agent_dummy = Agent(ax,ay)
    target_dummy = Agent(tx,ty)
    di = dist(agent_dummy,target_dummy)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent_dummy,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy) +5
    # print(cent_err)
    Ji = 0
    Ji += 0.5*((di-d0)**2)*(1 - cost_alpha)
    Ji += ((dc - r0)**2)*cost_alpha
    Ji += (1/n)*(cent_err**2)*(cost_alpha)
    
    return Ji   
def J_i_PG_int(ax,ay,angle_0,ang_ind,tx,ty,centroid,r0,d0,n): 
    test_angle = (ang_ind/na)*(2*np.pi)
    agent_dummy = Agent(ax,ay)
    target_dummy = Agent(tx,ty)
    di = dist(agent_dummy,target_dummy)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent_dummy,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy)
    Ji = 0
    Ji += 0.5*((di-d0)**2)*cost_alpha
    Ji += ((dc - r0)**2)*cost_alpha
    Ji += (1/n)*(cent_err**2)*(1 - cost_alpha)
    Ji += (0.5/TAU)*((angle_0 - test_angle )**2)
    return Ji   