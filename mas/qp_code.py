import cvxpy as cp
import numpy as np

# Define the variables
n = 2  # Number of variables
x = cp.Variable(n)

# Define the objective function and constraints
Q = np.array([[2, 1], [1, 2]])  # Quadratic coefficient matrix
c = np.array([-2, -3])          # Linear coefficient vector
A = np.array([[1, 1], [-1, 2]]) # Coefficient matrix for inequality constraints
b = np.array([1, 2])            # Right-hand side of inequality constraints

# Define the objective function
objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)

# Define the constraints
constraints = [A @ x <= b]

# Define the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the optimization problem
problem.solve()

# Print the results
print("Optimal value:", problem.value)
print("Optimal solution:")
print(x.value)
