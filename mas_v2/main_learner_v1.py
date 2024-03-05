from components.imports import *
from components.u_classes import *
from components.controller import *
from components.parameters import *
from components.functions import *
from tqdm import tqdm



read_q = "q_func_4.npy"
write_q = "q_func_4.npy"

global q_func
# q_func = np.zeros((nx,ny,nx,ny,nx,ny,na))
q_func = np.load(read_q)


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

def q_update(ind,history,new_indices):    
    # dummy_team = []
    # for a in ag:
    #     dummy_team.append(Agent(a.x,a.y))
    # for i in range(N):
    #     if i == N-1:
    #         dummy_team[i].target = dummy_team[0]
    #     else:
    #         dummy_team[i].target  = dummy_team[i+1]
    
    q_indices = history[ind]['indices'][-2]
    J0 = history[ind]['J0']
    J1 = history[ind]['J1']
    L = q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],q_indices[6]]
    N = q_func[new_indices[0],new_indices[1],new_indices[2],new_indices[3],new_indices[4],new_indices[5],new_indices[6]]
    q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],q_indices[6]]  = ((1 - learning_rate)*L) + (learning_rate*(J0 + (GAMMA*J1) + ((GAMMA**2)*N) ))
    return



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


# plt.ion()
# fig, ax = plt.subplots()
# plt.grid()

# # plot_data = ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],marker='o')
# plot_data, = ax.plot(np.array(traj)[:,0],np.array(traj)[:,1])

# # plt.ion()
# # fig, ax = plt.subplots()
# # plt.grid()

# # # plot_data = ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],marker='o')
# # plot_data, = ax.plot(np.array(traj)[:,0],np.array(traj)[:,1])

# plt.xlim(-15,15)
# plt.ylim(-15,15)
traj_cost = 0
episode = 0

traj_cost_list = []

for i in range(EPOCH):
    
    for episode in tqdm(range(EPISODES), desc= "Episode number : {} of {} | Trajectory cost = {:.2f} ".format(i,EPOCH,traj_cost)):
        traj_cost_list.append(traj_cost)
        traj_cost = 0
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
        history = []
        for i in range(N):
            history.append({
                "indices" : [],
                "J0" : 0,
                "J1" : 0
            })
        # print(history)
        while (collision):
            count = count + 1
            for ind in range(N):
                ax = ag[ind].x
                ay = ag[ind].y
                tx = ag[ind].target.x
                ty = ag[ind].target.y
                [cx,cy] = get_centroid(ag)
                J1 = J_i(ag[ind],[cx,cy],r0,d0,N)
                if count < 50 : traj_cost += J1
                history[ind]['J0'] = history[ind]['J1']
                history[ind]['J1'] = J1
                # print(history)
                temp_a = ag[ind]
                [cx,cy] = get_centroid(ag)
                temp_indices = get_q_index(temp_a.x,temp_a.y,temp_a.target.x,temp_a.target.y,cx,cy)
                temp_search = q_func[temp_indices[0],temp_indices[1],temp_indices[2],temp_indices[3],temp_indices[4],temp_indices[5],:]
                min_ind = np.argmin(temp_search)
                q_next_state(ag[ind],min_ind)
                temp_indices.append(min_ind)
                history[ind]['indices'].append(temp_indices)
                collision = check_collision(ind)
                if collision == 0:
                    q_func[temp_indices[0],temp_indices[1],temp_indices[2],temp_indices[3],temp_indices[4],temp_indices[5],min_ind] = 1000000
                    break
                temp_a = ag[ind]
                [cx,cy] = get_centroid(ag)
                temp_indices = get_q_index(temp_a.x,temp_a.y,temp_a.target.x,temp_a.target.y,cx,cy)
                temp_search = q_func[temp_indices[0],temp_indices[1],temp_indices[2],temp_indices[3],temp_indices[4],temp_indices[5],:]
                min_ind = np.argmin(temp_search)
                temp_indices.append(min_ind)
                new_indices = temp_indices
                
                if count >1:
                    q_update(ind,history,new_indices)
                
                # print("--------------",ind)
                # i = history[ind]['indices']
                # print(count,i,temp_indices.append(min_ind))
                
            

        
                
            for a in ag:
                traj.append([a.x,a.y,a.theta])
            # plot_data.set_offsets(np.c_[np.array(traj)[-4:,0],np.array(traj)[-4:,1]])
            # print(traj)


            
            # arr = np.array(traj)[-4:,0]
            # arr1 = np.array(traj)[-4:,1]
            # arr = np.concatenate((arr,np.array(traj)[-4:-3,0]))
            # arr1 = np.concatenate((arr1,np.array(traj)[-4:-3,1]))
            # plot_data.set_xdata(arr)
            # plot_data.set_ydata(arr1)
            
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # time.sleep(0.5)



            # fig.canvas.draw_idle()
            # plt.pause(0.001)
            if count > 200:
                break
    np.save(write_q,q_func)
    np.save("traj_cost_data_3",np.array(traj_cost_list))

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