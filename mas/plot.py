from components.imports import *



file_path = 'learn.txt'

# Lists to store coordinates
px = []
py = []
tx = []
ty = []

# Read coordinates from the file
with open(file_path, 'r') as file:
    for line in file:
        # Split the line into four values
        values = line.split()
        # Convert values to float and append to respective lists
        px.append(float(values[0]))
        py.append(float(values[1]))
        tx.append(float(values[2]))
        ty.append(float(values[3]))

# Plotting
plt.figure(figsize=(12, 12))
plt.scatter(px, py, color='green', label='P')
plt.scatter(tx, ty, color='red', label='T')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Coordinates')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()



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

# plt.show()
