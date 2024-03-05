import matplotlib.pyplot as plt
import numpy as np
import time

# Turn on interactive mode
plt.ion()

# Generate initial data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a line plot
fig, ax = plt.subplots()
line, = ax.plot(x, y, label='Sin(x)')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()

# Initialize five red points at random locations
num_points = 5
x_points = np.random.uniform(0, 10, num_points)
y_points = np.random.uniform(-1, 1, num_points)
points, = ax.plot(x_points, y_points, 'ro')  # 'ro' for red circles

# Display the initial plot
plt.show()

# Update the plot in a loop
for i in range(1, 10000):
    # Generate new data for the line plot update
    y_new = np.sin(x + i * 0.1)

    # Update the y-data of the line plot
    line.set_ydata(y_new)

    # Update the coordinates of the five red points
    x_points = np.random.uniform(0, 10, num_points)
    y_points = np.random.uniform(-1, 1, num_points)

    # Update the red points on the plot
    points.set_xdata(x_points)
    points.set_ydata(y_points)

    # Let Matplotlib update the plot automatically
    plt.pause(1.05)

# Turn off interactive mode to keep the plot window open
plt.ioff()
plt.show()
