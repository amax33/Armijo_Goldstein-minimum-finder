import numpy as np
import matplotlib.pyplot as plt


# Define the objective function
def objective_function(x):
    return x**3

# Armijo-Goldstein line search
def armijo_goldstein(func, x0, alpha=0.1, beta=0.5, max_iter=500, tol=1e-9):
    grad = lambda x: 3 * (x**2) # Analytical gradient
    iter_count = 0
    intermediate_points = [(x0, iter_count)]  # List to store intermediate points and iteration numbers
    while iter_count < max_iter:
        slope = grad(x0)
        step_size = alpha
        armijo_satisfied = False
        goldstein_satisfied = False
        while not (armijo_satisfied and goldstein_satisfied):
            if func(x0 - step_size * slope) <= func(x0) - beta * step_size * slope ** 2:
                armijo_satisfied = True
            if func(x0 - step_size * slope) >= (1 - beta) * func(x0) + beta * step_size * slope ** 2:
                goldstein_satisfied = True
            if not (armijo_satisfied and goldstein_satisfied):
                step_size *= beta
        x0 -= step_size * slope
        iter_count += 1
        intermediate_points.append((x0, iter_count))  # Store intermediate point and iteration number
        # Check convergence based on change in objective function value
        if iter_count > 1 and np.abs(func(intermediate_points[-1][0]) - func(intermediate_points[-2][0])) < tol:
            break
    return x0, intermediate_points

# Generate a random starting point
np.random.seed(0)
start_point = np.random.uniform(5, 15)

# Find the minimum using Armijo-Goldstein method
minimum_point, intermediate_points = armijo_goldstein(objective_function, start_point)

# Print the optimized point
print("Optimized point:", minimum_point)

# Plot the movement toward the answer
x_values = np.linspace(0, 15, 400)
y_values = objective_function(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Objective Function')
for point, iteration in intermediate_points:
    plt.scatter(point, objective_function(point), color='red')
    plt.text(point, objective_function(point), f'Iter: {iteration}', verticalalignment='bottom', horizontalalignment='right', fontsize=8)
plt.title('Movement Toward the Answer')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
