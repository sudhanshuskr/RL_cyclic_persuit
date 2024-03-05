from components.imports import *

window_size = 10
traj1 = np.load("traj_cost_data_dump.npy")
# traj2 = np.load("traj_cost_data_2.npy")[1:]
# traj3 = np.load("traj_cost_data_3.npy")[1:]
# traj = np.hstack((traj1,traj2,traj3))
traj = traj1[::100]

traj_new = np.zeros(traj.shape[0]-window_size +1)

for i in range(traj.shape[0]-window_size +1):
    traj_new[i] = (1/window_size)*(np.sum(traj[i:i+window_size]))

traj_new = np.hstack(([0],traj_new))




# print(traj_new.shape)
plt.figure(figsize=(25,10))
plt.title("Trajectory cost v/s episodes [BR , e = 0.5]",fontsize = 20)
plt.xlabel("Episodes",fontsize = 20)
plt.ylabel("Cost",fontsize = 20)
plt.plot(traj,alpha = 0.8)
plt.plot(traj_new,linewidth = 4,color = "darkblue")
plt.grid()

plt.show()
