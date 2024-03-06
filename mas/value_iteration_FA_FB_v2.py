from components.imports import *
from components.u_classes import *
from components.controller import *
from datetime import datetime
from components.functions import *
from tqdm import tqdm
from matplotlib.patches import Circle
from components.FAparameters import *

def move_agent(agent,vec):
    if np.linalg.norm(vec) > 1:
        vec = vec/(np.linalg.norm(vec))
    agent.x += vec[0]
    agent.y += vec[1]
    agent.theta = math.atan2(vec[1],vec[0])

def vec_to_direction(vec):
    return math.atan2(vec[1],vec[0])

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


def norm_state(states):
    states = np.array(states) + xlim
    states = states/(2*xlim)
    return states.tolist()
def norm_action(action):
    action = action + np.pi
    action = action/(2*np.pi)
    return action
def de_norm(states):
    states = states*(2*xlim)
    states = states - xlim
    return states

def Phi(states):
    states = norm_state(states)
    ans  = np.pi*np.matlum(C,states)
    ans = np.cos(ans)
    return ans

def safe_gradient_filter(ind,cx_0,cy_0):
    if ind == 0:
            l_ind = N-1
    else:
        l_ind = ind -1
    ax_0 = ag[ind].x
    ay_0 = ag[ind].y
    tx_0 = ag[ind].target.x
    ty_0 = ag[ind].target.y
    l_ax_0 = ag[l_ind].x
    l_ay_0 = ag[l_ind].y

    MIN_QS1 = 10000000
    MIN_QS1_ind = 0

    GRAD,gp,del_gp = J_grad_vec(l_ax_0,l_ay_0,ax_0,ay_0,tx_0,ty_0,[cx_0,cy_0],r0,d0,N,obs_x,obs_y)
    
    xi  = safe_grad(GRAD,gp,del_gp) 
    # print(np.linalg.norm(xi),np.linalg.norm(GRAD))

    return xi

def q_fa(S,A,wei,combin):
    S = norm_state(S)
    S.append(norm_action(A))
    temp = np.matmul(combin,np.array(S).reshape(n_states,1))
    temp = np.pi*temp
    temp = np.cos(temp)
    temp = temp.reshape(temp.shape[0],)
    temp = np.dot(temp,wei)
    
    return temp

def update_weights(S,A,W,combin,G):
    # W = weights vector
    # S = State vector 
    # A =  
    S = norm_state(S)
    # print(np.max(np.array(S)))
    S.append(norm_action(A))
    
    temp = np.matmul(combin,np.array(S).reshape(n_states,1))
    temp = np.pi*temp
    temp = np.cos(temp)
    q_old = q_fa(S[:-1],A,W,C)
    W = W + (learning_rate*(G - q_old)*temp).reshape(temp.shape[0],)
    return W


    


global v_func_w,C
v_func_w = np.zeros((n_weights))
combinations = product(range(FB_order+1), repeat=n_states)
C = []
for comb in combinations:
    C.append(comb)
C = np.array(C)


traj = [[0,0,0]]

if PLOT ==1 :
    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.grid()
    plot_data, = ax.plot(np.array(traj)[:,0],np.array(traj)[:,1])
    # point_data, = ax.plot(0, 0, 'ro') 
    cent_data, = ax.plot(0, 0, 'bo') 
    circle = Circle((obs_x, obs_y), radius=1, color='blue', alpha=0.5)  # (0, 0) represents the center of the circle, radius=1 defines the radius of the circle
    ax.add_patch(circle)
    plt.xlim(-15,15)    
    plt.ylim(-15,15)
traj_cost = 0
traj_cost_list = []
traj_cost_avg = 0
with open("save_dump/log.txt", 'a') as file:
    file.write(f"######################### Starting new run at : {datetime.now()}  ###############################\n")
Weights_track = []
for i in range(EPOCH):
    with open("save_dump/log.txt", 'a') as file:
            file.write(f"Epoch number {i} - Time : {datetime.now()}\n")
    for episode in range(EPISODES):
        
        REWARDS = []
        for ind in range(N):
            REWARDS.append([])
        for ind in range(N):
            REWARDS[ind].append(0)
        ACTIONS = []
        for ind in range(N):
            ACTIONS.append([])
        ACTIONS_VEC = []
        for ind in range(N):
            ACTIONS_VEC.append([])
        
        STATE = []
        for ind in range(N):
            STATE.append([])
        if episode == EPISODES -1:
            epsilon = EPSILON_TEST
            traj_cost = 0
            
        else:
            epsilon = EPSILON_TRAIN
        centroid_estimate = np.zeros((N,2))
        traj_cost_list.append(traj_cost)
        traj_cost_avg = np.average(traj_cost_list[-episode:])
        traj_cost = 0
        ag = []
        
        spawn_angle = random.uniform(0,2*np.pi)
        for n_agent in range(N):
            ag.append(Agent(spawn_center[0] + random.uniform(spawn_start,spawn_end)+ spawn_radius*np.cos(spawn_angle),spawn_center[1] + random.uniform(spawn_start,spawn_end)+ spawn_radius*np.sin(spawn_angle)))
            spawn_angle += (2*np.pi)/N
            if n_agent >=1:
                ag[n_agent-1].setTarget(ag[n_agent])
            if n_agent == N-1:
                ag[n_agent].setTarget(ag[0])
        
        for c in range(N):
            x_temp = ag[c].x
            y_temp = ag[c].y
            centroid_estimate[c] = [x_temp,y_temp]
        [cx,cy] = get_centroid(ag)
        for ind in range(N):
            STATE[ind].append([ag[ind].x,ag[ind].y,ag[ind].target.x,ag[ind].target.y,cx,cy])

        t = 0
        traj = [[0,0,0]]
        collision = 1
        count = 0
        
        while (collision):
            count = count + 1
            
            controls = []
            if t == 0:
                for ind in range(N):
                    [cx,cy] = get_centroid(ag)
                    update_vec = safe_gradient_filter(ind,cx,cy)
                    update_dir = vec_to_direction(update_vec)
                    ACTIONS[ind].append(update_dir)
                    ACTIONS_VEC[ind].append(update_vec)

            
            for ind in range(N):
                ax = ag[ind].x
                ay = ag[ind].y
                tx = ag[ind].target.x
                ty = ag[ind].target.y
                [cx,cy] = get_centroid(ag)



                move_agent(ag[ind],ACTIONS_VEC[ind][t])
                [cx,cy] = get_centroid(ag)
                STATE[ind].append(tuple([ag[ind].x,ag[ind].y,ag[ind].target.x,ag[ind].target.y,cx,cy]))
                
                collision = check_collision(ind)
                if collision == 0:
                    break
                REWARDS[ind].append(-J_i_int(ag[ind].x,ag[ind].y,ag[ind].target.x,ag[ind].target.y,[cx,cy],r0,d0,N))              
                update_vec = safe_gradient_filter(ind,cx,cy)
                update_dir = vec_to_direction(update_vec)
                ACTIONS[ind].append(update_dir)
                ACTIONS_VEC[ind].append(update_vec)
                tau = t - SARSA_n + 1
                if t%40 == 0:
                    Weights_track.append(v_func_w)
                if tau>=0:
                    final_T = np.minimum(EPISODE_LENGTH,tau + SARSA_n)
                    disc_time = tau+1
                    G = 0
                    while disc_time<=final_T:
                        G = G + (GAMMA**(disc_time-tau-1))*REWARDS[ind][disc_time]
                        disc_time += 1
                    if tau + SARSA_n < EPISODE_LENGTH:
                        G = G + ((GAMMA**SARSA_n)*q_fa(STATE[ind][t+1],ACTIONS[ind][t+1],v_func_w,C))
                    v_func_w = update_weights(STATE[ind][tau],ACTIONS[ind][tau],v_func_w,C,G)
                    if episode%40 == 0:
                        with open("save_dump/log.txt", 'a') as file:
                            file.write(f"Weight have been updated at Epoch : {i} and episode : {episode} - Time : {datetime.now()}\n")
                if tau >=EPISODE_LENGTH-1:
                    break
            [cx,cy] = get_centroid(ag)
            
            for a in ag:
                traj.append([a.x,a.y,a.theta])
            t = t+1
            

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
                # point_data.set_xdata(centroid_data_x)
                # point_data.set_ydata(centroid_data_y)
                cent_data.set_xdata(cx)
                cent_data.set_ydata(cy)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.005)



            if count > EPISODE_LENGTH:
                break
    if SAVE == 1:
        np.save("FA_weights_1.npy",np.array(Weights_track))
        with open("save_dump/log.txt", 'a') as file:
            file.write(f"Weights file has been saved @ Epoch {i} - Time : {datetime.now()}\n")
plt.waitforbuttonpress()

