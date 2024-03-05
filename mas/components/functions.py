from components.imports import *
from components.rk4 import *
from components.u_classes import *
import cvxpy as cp

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
    

def dist(a1,a2):
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
    Ji += ((di-d0)**2)*cost_alpha
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
    cent_err = dist(centroid_dummy,beacon_dummy) +10    
    # print(cent_err)
    Ji = 0
    Ji += ((di-d0)**2)*(1 - cost_alpha)
    Ji += ((dc - r0)**2)*cost_alpha
    Ji += (1/n)*(cent_err**2)*(cost_alpha)
    
    return Ji   
def J_i_PG_i_int(ax,ay,angle_0,ang_ind,tx,ty,centroid,r0,d0,n): 
    test_angle = (ang_ind/na)*(2*np.pi)
    agent_dummy = Agent(ax,ay)
    target_dummy = Agent(tx,ty)
    di = dist(agent_dummy,target_dummy)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent_dummy,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy)
    Ji = 0
    Ji += ((di-d0)**2)*cost_alpha
    Ji += ((dc - r0)**2)*cost_alpha
    Ji += (1/n)*(cent_err**2)*(1 - cost_alpha)
    Ji += (0.5/TAU)*((angle_0 - test_angle )**2)
    print((0.5/TAU)*((angle_0 - test_angle )**2),Ji - (0.5/TAU)*((angle_0 - test_angle )**2),Ji)
    return Ji   

def adjust_angle(theta):
    if theta >=2*np.pi:
        theta = theta - 2*np.pi
    elif theta < 0:
        theta = theta + 2*np.pi

def J_i_PG_int(lax,lay,ax,ay,angle_0,ang_ind,tx,ty,centroid,r0,d0,n): 
    test_angle = (ang_ind/na)*(2*np.pi)
    if test_angle < angle_0:
        theta_diff = angle_0 - test_angle        
    else:
        theta_diff = test_angle - angle_0
    if theta_diff > np.pi:
        theta_diff = 2*np.pi - theta_diff

    last_agent_dummy = Agent(lax,lay)
    agent_dummy = Agent(ax,ay)
    target_dummy = Agent(tx,ty)
    l_di = dist(last_agent_dummy,agent_dummy)
    di = dist(agent_dummy,target_dummy)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent_dummy,centroid_dummy)
    l_dc = dist(last_agent_dummy,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy)
    Ji = 0
    Ji += ((di-d0)**2)*cost_alpha
    Ji += ((l_di-d0)**2)*cost_alpha
    Ji += ((dc - r0)**2)*cost_alpha
    Ji += ((l_dc - r0)**2)*cost_alpha
    Ji += (1/n)*(cent_err**2)*(1 - cost_alpha)+4000
    Ji += (0.5/TAU)*((theta_diff)**2)
    # print((0.5/TAU)*((angle_0 - test_angle )**2),Ji - (0.5/TAU)*((angle_0 - test_angle )**2),Ji)
    return Ji   

def grad(lax,lay,ax,ay,angle_0,ang_ind,tx,ty,centroid,r0,d0,n): 
    test_angle = (ang_ind/na)*(2*np.pi)
    if test_angle < angle_0:
        theta_diff = angle_0 - test_angle        
    else:
        theta_diff = test_angle - angle_0
    if theta_diff > np.pi:
        theta_diff = 2*np.pi - theta_diff

    last_agent_dummy = Agent(lax,lay)
    agent_dummy = Agent(ax,ay)
    target_dummy = Agent(tx,ty)
    l_di = dist(last_agent_dummy,agent_dummy)
    di = dist(agent_dummy,target_dummy)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent_dummy,centroid_dummy)
    l_dc = dist(last_agent_dummy,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy)
    Gx = 2*(di-d0)*(1/di)*(agent_dummy.x - target_dummy.x)
    Gx +=  2*(dc-r0)*(1/dc)*(agent_dummy.x - centroid_dummy.x)
    Gx += 20*(centroid_dummy.x - beacon_dummy.x)*(1/n)*(1/n)
    Gy = 2*(di-d0)*(1/di)*(agent_dummy.y - target_dummy.y)
    Gy +=  2*(dc-r0)*(1/dc)*(agent_dummy.y - centroid_dummy.y)
    Gy += 20*(centroid_dummy.y - beacon_dummy.y)*(1/n)*(1/n)

    Gx += 2*(l_di-d0)*(1/l_di)*(agent_dummy.x - last_agent_dummy.x)
    Gx += 20*(centroid_dummy.x - beacon_dummy.x)*(1/n)*(1/n)
    Gy += 2*(l_di-d0)*(1/l_di)*(agent_dummy.y - last_agent_dummy.y)
    Gy += 20*(centroid_dummy.y - beacon_dummy.y)*(1/n)*(1/n)
    if (np.isclose(Gx,0,atol=1) and np.isclose(Gy,0,atol = 1)):
        GRAD = 0 #handle 0 vector
        rad_vec = np.array([agent_dummy.x - centroid_dummy.x,agent_dummy.y-centroid_dummy.y,0])
        rad_vec = np.cross(np.array([0,0,1]),rad_vec)
        GRAD = np.arctan2(rad_vec[1],rad_vec[0])

    else:
        GRAD = np.arctan2(Gy,Gx) + np.pi
        if GRAD < 0:
            GRAD += 2*np.pi
    return GRAD

def grad_vec(lax,lay,ax,ay,angle_0,ang_ind,tx,ty,centroid,r0,d0,n): 
    test_angle = (ang_ind/na)*(2*np.pi)
    if test_angle < angle_0:
        theta_diff = angle_0 - test_angle     
    else:
        theta_diff = test_angle - angle_0
    if theta_diff > np.pi:
        theta_diff = 2*np.pi - theta_diff

    last_agent_dummy = Agent(lax,lay)
    agent_dummy = Agent(ax,ay)
    target_dummy = Agent(tx,ty)
    l_di = dist(last_agent_dummy,agent_dummy)
    di = dist(agent_dummy,target_dummy)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent_dummy,centroid_dummy)
    l_dc = dist(last_agent_dummy,centroid_dummy)
    cent_err = dist(centroid_dummy,beacon_dummy)
    Gx = 2*(di-d0)*(1/di)*(agent_dummy.x - target_dummy.x)
    Gx +=  2*(dc-r0)*(1/dc)*(agent_dummy.x - centroid_dummy.x)
    Gx += 2*(centroid_dummy.x - beacon_dummy.x)*(1/n)*(1/n)
    Gy = 2*(di-d0)*(1/di)*(agent_dummy.y - target_dummy.y)
    Gy +=  2*(dc-r0)*(1/dc)*(agent_dummy.y - centroid_dummy.y)
    Gy += 2*(centroid_dummy.y - beacon_dummy.y)*(1/n)*(1/n)

    Gx += 2*(l_di-d0)*(1/l_di)*(agent_dummy.x - last_agent_dummy.x)
    Gx += 2*(centroid_dummy.x - beacon_dummy.x)*(1/n)*(1/n)
    Gy += 2*(l_di-d0)*(1/l_di)*(agent_dummy.y - last_agent_dummy.y)
    Gy += 2*(centroid_dummy.y - beacon_dummy.y)*(1/n)*(1/n)
    return [Gx,Gy],(L**2 - di**2),[2*(agent_dummy.x - target_dummy.x),2*(agent_dummy.y - target_dummy.y)]
    
    if (np.isclose(Gx,0,atol=1) and np.isclose(Gy,0,atol = 1)):
        GRAD = 0 #handle 0 vector
        rad_vec = np.array([agent_dummy.x - centroid_dummy.x,agent_dummy.y-centroid_dummy.y,0])
        rad_vec = np.cross(np.array([0,0,1]),rad_vec)
        GRAD = np.arctan2(rad_vec[1],rad_vec[0])

    else:
        GRAD = np.arctan2(Gy,Gx) + np.pi
        if GRAD < 0:
            GRAD += 2*np.pi
    return GRAD
def J_grad_vec(lax,lay,ax,ay,tx,ty,centroid,r0,d0,n,ox,oy): 
    last_agent_dummy = Agent(lax,lay)
    agent_dummy = Agent(ax,ay)
    target_dummy = Agent(tx,ty)
    obstacle_dummy = Agent(ox,oy)
    l_di = dist(last_agent_dummy,agent_dummy)
    di = dist(agent_dummy,target_dummy)
    do = dist(agent_dummy,obstacle_dummy)
    centroid_dummy = Agent(centroid[0],centroid[1])
    beacon_dummy = Agent(0,0)
    dc = dist(agent_dummy,centroid_dummy)
    Gx = 2*(di-d0)*(1/di)*(agent_dummy.x - target_dummy.x)
    Gx +=  2*(dc-r0)*(1/dc)*(agent_dummy.x - centroid_dummy.x)
    Gx += 4*(centroid_dummy.x - beacon_dummy.x)*(1/n)*(1/n)
    Gy = 2*(di-d0)*(1/di)*(agent_dummy.y - target_dummy.y)
    Gy +=  2*(dc-r0)*(1/dc)*(agent_dummy.y - centroid_dummy.y)
    Gy += 4*(centroid_dummy.y - beacon_dummy.y)*(1/n)*(1/n)

    Gx += 2*(l_di-d0)*(1/l_di)*(agent_dummy.x - last_agent_dummy.x)
    Gx += 4*(centroid_dummy.x - beacon_dummy.x)*(1/n)*(1/n)
    Gy += 2*(l_di-d0)*(1/l_di)*(agent_dummy.y - last_agent_dummy.y)
    Gy += 4*(centroid_dummy.y - beacon_dummy.y)*(1/n)*(1/n)
    return (0.1/np.linalg.norm(np.array([Gx,Gy])))*np.array([Gx,Gy]),(L**2 - do**2),[2*(-agent_dummy.x + obstacle_dummy.x),2*(-agent_dummy.y + obstacle_dummy.y)]
    
    if (np.isclose(Gx,0,atol=1) and np.isclose(Gy,0,atol = 1)):
        GRAD = 0 #handle 0 vector
        rad_vec = np.array([agent_dummy.x - centroid_dummy.x,agent_dummy.y-centroid_dummy.y,0])
        rad_vec = np.cross(np.array([0,0,1]),rad_vec)
        GRAD = np.arctan2(rad_vec[1],rad_vec[0])

    else:
        GRAD = np.arctan2(Gy,Gx) + np.pi
        if GRAD < 0:
            GRAD += 2*np.pi
    return GRAD
    
def safe_grad(GRAD,gp,del_gp):
    n = 2  # Number of variables
    x = cp.Variable(n)

    GRAD = np.array(GRAD)
    # Define the objective function and constraints
    Q = np.array([[2, 0], [0, 2]])  # Quadratic coefficient matrix
    c = np.array(2*GRAD)          # Linear coefficient vector
    A = np.array(del_gp) # Coefficient matrix for inequality constraints
    b = np.array(-alpha*gp)            # Right-hand side of inequality constraints

    # Define the objective function
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)

    # Define the constraints
    constraints = [A @ x <= b]

    # Define the optimization problem
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem
    problem.solve()
    return x.value