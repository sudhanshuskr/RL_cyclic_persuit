from components.parameters import *
from components.imports import *
# def rk4_step(f,x,t,dt,n):
#     k1 = f(t,x,n)
#     k2 = f(t + (0.5*dt),x + (0.5*k1*dt),n)
#     k3 = f(t + (0.5*dt),x + (0.5*k2*dt),n)
#     k4 = f(t + dt,x + (k3*dt),n)

#     return dt*((k1 + (2*k2) + (2*k3) + k4)/6)

class Agent_rk:
    def __init__(self, x, y, theta = 0, name = "untitled"):
        self.x = x
        self.y = y
        self.name = str(name)
        self.theta = theta
        self.target = None
    def setTarget(self,ag):
        self.target = ag
    def __str__(self):
      return f"Name : {self.name} |:| Coordinates : ({self.x},{self.y}) |:| Heading : {self.theta} radians"
    
    
def rk4_step(f,team,t,controller):
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    
    for agent in team:
        k1.append(f(agent,t,controller))
    
    i = 0
    # print("k1 done")
    for agent in team :
        x = agent.x
        y = agent.y
        theta = agent.theta
        agent_temp = Agent_rk(x + (0.5*k1[i][0]*dt),y + (0.5*k1[i][1]*dt),theta + (0.5*k1[i][2]*dt),name=agent.name)
        agent_temp.target = agent.target
        k2.append(f(agent_temp,t + (0.5*dt),controller))
        i = i + 1 
    i = 0
    for agent in team:
        x = agent.x
        y = agent.y
        theta = agent.theta
        agent_temp = Agent_rk(x + (0.5*k2[i][0]*dt),y + (0.5*k2[i][1]*dt),theta + (0.5*k2[i][2]*dt),name=agent.name)
        agent_temp.target = agent.target
        k3.append(f(agent_temp,t + (0.5*dt),controller))
        i = i + 1 
    i = 0
    for agent in team:
        x = agent.x
        y = agent.y
        theta = agent.theta
        agent_temp = Agent_rk(x + (k3[i][0]*dt),y + (k3[i][1]*dt),theta + (k3[i][2]*dt),name=agent.name)
        agent_temp.target = agent.target
        k4.append(f(agent_temp,t + dt,controller))
        i = i+1
    
    k1 = np.array(k1)
    k2 = np.array(k2)
    k3 = np.array(k3)
    k4 = np.array(k4)

    return dt*((k1 + (2*k2) + (2*k3) + k4)/6)