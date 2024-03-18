from components.imports import *
from components.u_classes import *
from components.controller import *
from components.parameters import *
from components.functions import *
from tqdm import tqdm





def neighbour_grid(i):
    angle = i*(np.pi/4)
    dx = int(np.ceil(np.cos(angle)))
    dy = int(np.ceil(np.sin(angle)))
    return (dx,dy)

global v_func
v_func = np.zeros((nx,ny,nx,ny,nx,ny))
global a_func
a_func = np.zeros((nx,ny,nx,ny,nx,ny))


time_horizon = 100
rev_t = 0
for rev_t in tqdm(range(time_horizon)):
    for a1_x_ind in range(nx):
        for a1_y_ind in range(ny):
            for a2_x_ind in range(nx):
                for a2_y_ind in range(ny):
                    for a3_x_ind in range(nx):
                        for a3_y_ind in range(ny):
                            theta_check_i = []
                            min_action = "aaa"
                            for theta1 in range(8):
                                for theta2 in range(8):
                                    for theta3 in range(8):
                                        (dx1,dy1) = neighbour_grid(theta1)
                                        (dx2,dy2) = neighbour_grid(theta1)
                                        (dx3,dy3) = neighbour_grid(theta1)
                                        a1_xp_ind = a1_x_ind + dx1
                                        a1_yp_ind = a1_y_ind + dy1
                                        a2_xp_ind = a2_x_ind + dx2
                                        a2_yp_ind = a2_y_ind + dy2
                                        a3_xp_ind = a3_x_ind + dx3
                                        a3_yp_ind = a3_y_ind + dy3
                                        if max([a1_xp_ind,a2_xp_ind,a3_xp_ind]) >= nx or max([a1_yp_ind,a2_yp_ind,a3_yp_ind]) >= ny:
                                            continue
                                        if min([a1_xp_ind,a2_xp_ind,a3_xp_ind]) < 0 or min([a1_yp_ind,a2_yp_ind,a3_yp_ind]) < 0:
                                            continue 
                                        next_check = np.array([a1_xp_ind,a1_yp_ind,a2_xp_ind,a2_yp_ind,a3_xp_ind,a3_yp_ind])
                                        action_value = incur(next_check) + GAMMA*v_func[a1_xp_ind,a1_yp_ind,a2_xp_ind,a2_yp_ind,a3_xp_ind,a3_yp_ind]
                                        if not theta_check_i:
                                            theta_check_i.append(theta3)
                                            min_action_value = action_value
                                            min_action = str(theta1) + str(theta2) + str(theta3) 
                                        if action_value < min_action_value:
                                            min_action_value = action_value
                                            min_action = str(theta1) + str(theta2) + str(theta3)

                            a_func[a1_x_ind,a1_y_ind,a2_x_ind,a2_y_ind,a3_x_ind,a3_y_ind] = min_action
                            v_func[a1_x_ind,a1_y_ind,a2_x_ind,a2_y_ind,a3_x_ind,a3_y_ind] = min_action_value
    if rev_t%10 == 0:
        if SAVE == 1:
            np.save("v_func_dump.npy",v_func)
            np.save("a_func_dump.npy",a_func)                        

