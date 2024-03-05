from components.imports import *
from components.u_classes import *
from components.controller import *
from components.parameters import *
from components.functions import *
from tqdm import tqdm


global q_func
q_func = np.zeros((nx,ny,nx,ny,nx,ny,na))
# q_func = np.load('q_func_dump.npy')


def q_next_state(agent,theta_ind):
    theta_i = (theta_ind/na)*(2*np.pi)
    agent.x += v*np.cos(theta_i)
    agent.y += v*np.sin(theta_i)
    agent.theta = theta_i

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

def get_next_step(temp_indices):
    

    temp_search = q_func[temp_indices[0],temp_indices[1],temp_indices[2],temp_indices[3],temp_indices[4],temp_indices[5],:]
    min_ind = np.argmin(temp_search)
    return min_ind


def get_next_step_BR(ind,cx_0,cy_0):

    ax_0 = ag[ind].x
    ay_0 = ag[ind].y
    tx_0 = ag[ind].target.x
    ty_0 = ag[ind].target.y
    q_indices = get_q_index(ax_0,ay_0,tx_0,ty_0,cx_0,cy_0)
    J0 = J_i(ag[ind],[cx_0,cy_0],r0,d0,N)

    MIN_QS1 = 10000000
    MIN_QS1_ind = 0
    for i in range(na):
        theta_0 = (i/na)*(2*np.pi)
        ax_1 = ax_0 + (v*np.cos(theta_0))
        ay_1 = ay_0 + (v*np.sin(theta_0))
        cx_1 = cx_0 + ((1/N)*(v*np.cos(theta_0)))
        cy_1 = cy_0 + ((1/N)*(v*np.sin(theta_0)))
        if abs(ax_1) >= 0.98*xlim:
            q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],i] = 1000000
            continue
        if abs(ay_1) >= 0.98*ylim:
            q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],i] = 1000000
            continue
        J1 = J_i_int(ax_1,ay_1,tx_0,ty_0,[cx_1,cy_1],r0,d0,N)
        MIN_QS2 = 1000000
        for k in range(na):
            theta_1 = (k/na)*(2*np.pi)
            ax_2 = ax_1 + (v*np.cos(theta_1))
            ay_2 = ay_1 + (v*np.sin(theta_1))
            cx_2 = cx_1 + ((1/N)*(v*np.cos(theta_1)))
            cy_2 = cy_1 + ((1/N)*(v*np.sin(theta_1)))
            if abs(ax_2) >= 0.98*xlim:
                continue
            if abs(ay_2) >= 0.98*ylim:
                continue
            q_indices_s2 = get_q_index(ax_2,ay_2,tx_0,ty_0,cx_2,cy_2)
            theta_2_search = q_func[q_indices_s2[0],q_indices_s2[1],q_indices_s2[2],q_indices_s2[3],q_indices_s2[4],q_indices_s2[5],:]
            theta_2_ind = np.argmin(theta_2_search)
            q_val_s2 = q_func[q_indices_s2[0],q_indices_s2[1],q_indices_s2[2],q_indices_s2[3],q_indices_s2[4],q_indices_s2[5],theta_2_ind]
            if q_val_s2 < MIN_QS2:
                MIN_QS2 = q_val_s2
        q_val_s1 = J1 + (GAMMA*MIN_QS2)
        if q_val_s1 < MIN_QS1 :
            MIN_QS1 = q_val_s1
            MIN_QS1_ind = i
    
    rn = random.random()
    if rn < epsilon:
        pass
    else:
        MIN_QS1_ind = random.randint(0, na-1)

    return MIN_QS1_ind

def q_update(ind,history,new_indices):    
    q_indices = history[ind]['indices'][-2]
    J0 = history[ind]['J0']
    J1 = history[ind]['J1']
    L = q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],q_indices[6]]
    N = q_func[new_indices[0],new_indices[1],new_indices[2],new_indices[3],new_indices[4],new_indices[5],new_indices[6]]
    q_func[q_indices[0],q_indices[1],q_indices[2],q_indices[3],q_indices[4],q_indices[5],q_indices[6]]  = ((1 - learning_rate)*L) + (learning_rate*(J0 + (GAMMA*J1) + ((GAMMA**2)*N) ))
    return


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
if PLOT ==1 :
    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.grid()

    # plot_data = ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],marker='o')
    plot_data, = ax.plot(np.array(traj)[:,0],np.array(traj)[:,1])
    point_data, = ax.plot(0, 0, 'ro') 
    cent_data, = ax.plot(0, 0, 'bo') 
    plt.xlim(-15,15)    
    plt.ylim(-15,15)
traj_cost = 0
traj_cost_list = []
traj_cost_avg = 0
for i in range(EPOCH):
    for episode in tqdm(range(EPISODES), desc= "Episode number : {} of {} | Trajectory cost = {:.2f} ".format(i,EPOCH,traj_cost)):
        if episode == EPISODES -1:
            epsilon = EPSILON_TEST
            traj_cost = 0
            
        else:
            epsilon = EPSILON_TRAIN
        centroid_estimate = np.zeros((N,2))
        traj_cost_list.append(traj_cost)
        # print(traj_cost)
        traj_cost_avg = np.average(traj_cost_list[-episode:])
        traj_cost = 0
        ag = []
        
        temp_radius = 2
        spawn_angle = 0
        for n_agent in range(N):
            ag.append(Agent(spawn_center[0] + temp_radius*np.cos(spawn_angle),spawn_center[1] + temp_radius*np.sin(spawn_angle)))
            spawn_angle += (2*np.pi)/N
            if n_agent >=1:
                ag[n_agent-1].setTarget(ag[n_agent])
            if n_agent == N-1:
                ag[n_agent].setTarget(ag[0])
        
        for c in range(N):
            x_temp = ag[c].x
            y_temp = ag[c].y
            centroid_estimate[c] = [x_temp,y_temp]
        

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
            controls = []
            for ind in range(N):
                ax = ag[ind].x
                ay = ag[ind].y
                tx = ag[ind].target.x
                ty = ag[ind].target.y
                [cx,cy] = get_centroid(ag)
                J1 = J_i(ag[ind],[cx,cy],r0,d0,N)
                traj_cost += J1
                history[ind]['J0'] = history[ind]['J1']
                history[ind]['J1'] = J1
                # print(history)
                [cx,cy] = get_centroid(ag)
                temp_a = ag[ind]
                temp_indices = get_q_index(temp_a.x,temp_a.y,temp_a.target.x,temp_a.target.y,cx,cy)
                min_ind = get_next_step_BR(ind,centroid_estimate[c][0],centroid_estimate[c][1])
                if episode == EPISODES-1:
                    temp_search = q_func[temp_indices[0],temp_indices[1],temp_indices[2],temp_indices[3],temp_indices[4],temp_indices[5],:]
                    min_ind = np.argmin(temp_search)
                if random.random() > epsilon:
                    min_ind = random.randint(0,na-1)
                
                q_next_state(ag[ind],min_ind)
                controls.append(min_ind)

                ag_i = ind       
                fi = ag_i
                si = ag_i +1
                if fi == N-1: si = 0

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

            [cx,cy] = get_centroid(ag)
                
            for a in ag:
                traj.append([a.x,a.y,a.theta])
            if collision != 0:    
                for c in range(N):
                    if c == N - 1:
                        centroid_estimate[c][0] = 0.5*(centroid_estimate[c][0] + centroid_estimate[0][0]) + (v*np.cos((controls[c]/na)*(2*np.pi)))
                        centroid_estimate[c][1] = 0.5*(centroid_estimate[c][1] + centroid_estimate[0][1]) + (v*np.sin((controls[c]/na)*(2*np.pi)))
                        break
                    centroid_estimate[c][0] = 0.5*(centroid_estimate[c][0] + centroid_estimate[c+1][0]) + (v*np.cos((controls[c]/na)*(2*np.pi)))
                    centroid_estimate[c][1] = 0.5*(centroid_estimate[c][1] + centroid_estimate[c+1][1]) + (v*np.sin((controls[c]/na)*(2*np.pi)))


            if PLOT == 1:
                arr = np.array(traj)[-N:,0]
                arr1 = np.array(traj)[-N:,1]
                arr = np.concatenate((arr,np.array(traj)[-N:-N+1,0]))
                arr1 = np.concatenate((arr1,np.array(traj)[-N:-N+1,1]))
                plot_data.set_xdata(arr)
                plot_data.set_ydata(arr1)
                
                centroid_data_x = []
                centroid_data_y = []
                for c in range(N):
                    centroid_data_x.append(centroid_estimate[c][0])
                    centroid_data_y.append(centroid_estimate[c][1])
                point_data.set_xdata(centroid_data_x)
                point_data.set_ydata(centroid_data_y)
                cent_data.set_xdata(cx)
                cent_data.set_ydata(cy)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.005)



            if count > EPISODE_LENGTH:
                break
    if SAVE == 1:
        np.save("q_func_dump.npy",q_func)
        np.save("traj_cost_data_dump",np.array(traj_cost_list))

plt.waitforbuttonpress()

