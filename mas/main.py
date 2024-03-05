from components.imports import *
from components.u_classes import *
from components.controller import *
from components.parameters import *
from components.functions import *
from tqdm import tqdm



global q_func
q_func = np.zeros((nx,ny,nx,ny,nx,ny,na))
# q_func = np.load('q_func_3_13.npy')


def q_next_state(agent,theta_ind):
    theta_i = (theta_ind/na)*(2*np.pi)
    agent.x += v*np.cos(theta_i)
    agent.y += v*np.sin(theta_i)

def check_collision(index):
    agent = ag[index]
    if abs(agent.x) >=xlim:
        return 0
    elif abs(agent.y) >=ylim:
        return 0
    else:
        return 1


def get_temp_lin_agent(a,theta):
    return Agent(a.x + (v*np.cos(theta)),a.y + (v*np.sin(theta)))

def q_update(ind):    
    # dummy_team = []
    # for a in ag:
    #     dummy_team.append(Agent(a.x,a.y))
    # for i in range(N):
    #     if i == N-1:
    #         dummy_team[i].target = dummy_team[0]
    #     else:
    #         dummy_team[i].target  = dummy_team[i+1]
        
    ax = ag[ind].x
    ay = ag[ind].y
    tx = ag[ind].target.x
    ty = ag[ind].target.y
    [cx,cy] = get_centroid(ag)
    q_indices = get_q_index(ax,ay,tx,ty,cx,cy)
    Ji = J_i(ag[ind],[cx,cy],r0,d0,N)

    
    
    for i in range(na):
        theta_try = (i/na)*(2*np.pi)
        ax_update = ax + (v*np.cos(theta_try))
        ay_update = ay + (v*np.sin(theta_try))
        cx_update = cx + ((1/N)*(v*np.cos(theta_try)))
        cy_update = cy + ((1/N)*(v*np.sin(theta_try)))
        
        if abs(ax_update) >= 0.98*xlim:
            continue
        elif abs(ay_update) >= 0.98*ylim:
            continue
        else:
            MIN_ind = i
            break
    
    theta_try = (MIN_ind/na)*(2*np.pi)
    ax_update = ax + (v*np.cos(theta_try))
    ay_update = ay + (v*np.sin(theta_try))
    cx_update = cx + ((1/N)*(v*np.cos(theta_try)))
    cy_update = cy + ((1/N)*(v*np.sin(theta_try)))

    q_indices_update = get_q_index(ax_update,ay_update,tx,ty,cx_update,cy_update)
    


    temp_search = q_func[q_indices_update[0],q_indices_update[1],q_indices_update[2],q_indices_update[3],q_indices_update[4],q_indices_update[5],:]
    min_ind = np.argmin(temp_search)
    MIN = Ji + (GAMMA*q_func[q_indices_update[0],q_indices_update[1],q_indices_update[2],q_indices_update[3],q_indices_update[4],q_indices_update[5],min_ind]) 

    for i in range(na):
        theta_try = (i/na)*(2*np.pi)
        ax_update = ax + (v*np.cos(theta_try))
        ay_update = ay + (v*np.sin(theta_try))
        cx_update = cx + ((1/N)*(v*np.cos(theta_try)))
        cy_update = cy + ((1/N)*(v*np.sin(theta_try)))
        if abs(ax_update) >= 0.98*xlim:
            continue
        if abs(ay_update) >= 0.98*ylim:
            continue
        q_indices_update = get_q_index(ax_update,ay_update,tx,ty,cx_update,cy_update)
        temp_search = q_func[q_indices_update[0],q_indices_update[1],q_indices_update[2],q_indices_update[3],q_indices_update[4],q_indices_update[5],:]
        
        min_ind = np.argmin(temp_search)
        
        if MIN > (Ji + (GAMMA*q_func[q_indices_update[0],q_indices_update[1],q_indices_update[2],q_indices_update[3],q_indices_update[4],q_indices_update[5],min_ind])):
            MIN = Ji + (GAMMA*q_func[q_indices_update[0],q_indices_update[1],q_indices_update[2],q_indices_update[3],q_indices_update[4],q_indices_update[5],min_ind])
            MIN_ind = i

    rn = random.random()
    if rn < epsilon:
        pass
    else:
        MIN_ind = random.randint(0, na-1)
    L  = q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],MIN_ind]
    print(MIN - Ji)
    q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],MIN_ind] = (1 - learning_rate)*L + (learning_rate*(MIN))
    # print(q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],MIN_ind])
    return MIN_ind



global a1,a2,a3,a4,ag
a1 = Agent(-10,-10,0,name="1")
a2 = Agent(-8,-10,0,name="2")
a3 = Agent(-8,-8,0,name="3")
a4 = Agent(-10,-8,0,name="4")
a1.setTarget(a2)
a2.setTarget(a3)
a3.setTarget(a4)
a4.setTarget(a1)




ag = [a1,a2,a3,a4]
t = 0
traj = [[0,0,0]]


plt.ion()
fig, ax = plt.subplots()
plot_data = ax.scatter(np.array(traj)[:,0],np.array(traj)[:,1],marker='o',s=4)

plt.xlim(-15,15)
plt.ylim(-15,15)

for i in range(100):
    
    for _ in tqdm(range(EPISODES)):
        a1 = Agent(-10,-10,0,name="1")
        a2 = Agent(-8,-10,0,name="2")
        a3 = Agent(-8,-8,0,name="3")
        a4 = Agent(-10,-8,0,name="4")
        a1.setTarget(a2)
        a2.setTarget(a3)
        a3.setTarget(a4)
        a4.setTarget(a1)
        ag = [a1,a2,a3,a4]
        t = 0
        traj = [[0,0,0]]
        collision = 1
        count = 0
        while (collision):
            count = count + 1
            for ind in range(N):
                min_ind = q_update(ind)
                q_next_state(ag[ind],min_ind)
                collision = check_collision(ind)
                if collision == 0:
                    break
            for a in ag:
                traj.append([a.x,a.y,a.theta])
            plot_data.set_offsets(np.c_[np.array(traj)[-1,0],np.array(traj)[-1,1]])
            print(traj)
    
            fig.canvas.draw_idle()
            plt.pause(5)
            if count > 200:
                break
    np.save("q_func_3_13.npy",q_func)

plt.waitforbuttonpress()




# ag = [a1,a2,a3,a4]
# t = 0
# traj = [[0,0,0]]


# plt.ion()
# fig, ax = plt.subplots()
# plot_data = ax.scatter(np.array(traj)[:,0],np.array(traj)[:,1],marker='.',s=4)

# plt.xlim(-15,15)
# plt.ylim(-15,15)

# for i in tqdm(range(10)):
#     nextState(ag,f_linear,lin_controller,t)
#     t = t +dt
#     # for aaaa in ag:
#         # if t< 0.08:
#             # print(aaaa)
        
    
#     for a in ag:
#         traj.append([a.x,a.y,a.theta])
#     plot_data.set_offsets(np.c_[np.array(traj)[:,0],np.array(traj)[:,1]])
    
#     fig.canvas.draw_idle()
#     plt.pause(0.1)
# plt.waitforbuttonpress()


# plt.ion()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plot_data = ax.scatter(np.array(traj)[:,0],np.array(traj)[:,1],marker='.',s=1)
# for i in tqdm(range(400)):
#     nextState(ag,f_linear,lin_controller,t)
#     t = t +dt
#     # for aaaa in ag:
#         # if t< 0.08:
#             # print(aaaa)
        
    
#     for a in ag:
#         traj.append([a.x,a.y,a.theta])
    
# print(np.array(traj).shape)
# plt.scatter(np.array(traj)[:,0],np.array(traj)[:,1],marker='.',s=1)
# # plt.scatter(np.array(traj)[-10000:,0],np.array(traj)[-10000:,1],marker='.',s=1)
# # plt.plot(np.array(traj)[:,2],"*")
# plt.axis("equal")
# plt.show()




# plt.ion()
# fig, ax = plt.subplots()
# x, y = [],[]
# sc = ax.scatter(x,y)
# plt.xlim(0,10)
# plt.ylim(0,10)

# plt.draw()
# for i in range(10):
#     x.append(np.random.rand(1)*10)
#     y.append(np.random.rand(1)*10)
#     sc.set_offsets(np.c_[x,y])
#     fig.canvas.draw_idle()
#     plt.pause(0.1)

# plt.waitforbuttonpress()