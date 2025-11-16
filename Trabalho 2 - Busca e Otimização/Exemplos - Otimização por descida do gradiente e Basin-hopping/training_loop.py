# Demonstration of a training loop for linear regression using:
# - gradient descent
# - mini-batch gradient descent
# - stochastic gradient descent
# 
# The optimizer can run for 'max_iter' iterations or until convergence.
# The training data is separated in batches of 'batch_size' examples each.
# An epoch is a full pass through the training data for samples/batch_size times.
#
# Daniel Cavalcanti Jeronymo danielc@utfpr.edu.br 2025


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from optimizers import sgd

# Generate synthetic data for linear regression
rns = 0
np.random.seed(rns)

f_counter = 0
df_counter = 0

# Define loss function (Mean Squared Error) for batch
def mse_loss(params, X_batch, y_batch):
    global f_counter
    f_counter += len(X_batch) # count the number of samples in this batch for MSE
    
    w = params[:-1]
    b = params[-1]
    y_pred = X_batch @ w + b
    return np.mean((y_pred - y_batch) ** 2)

# Define gradient function for batch
def mse_grad(params, X_batch, y_batch):
    global df_counter
    df_counter += len(X_batch) # count the number of samples in this batch for MSE gradient
    
    w = params[:-1]
    b = params[-1]
    y_pred = X_batch @ w + b
    n = len(y_batch)
    
    # Gradient for weights
    dw = 2 * X_batch.T @ (y_pred - y_batch) / n
    
    # Gradient for bias
    db = 2 * np.sum(y_pred - y_batch) / n
    
    return np.concatenate([dw, [db]])

# Runs SGD for all batches in multiple epochs
# Each epoch processes all batches in a random order
def train_with_epochs(n_epochs, X, y, n_batches, learning_rate=0.01, max_iter=100):
    # Initial parameters (initialized to zeros)
    params = np.zeros(X.shape[1] + 1)  # +1 for bias
    
    # History for plotting
    epoch_losses = []
    all_losses = []
    all_params = [params.copy()]

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        # Shuffle batches for each epoch
        shuffled_idx = np.random.permutation(len(X))
        X_batches_shuffled = np.array_split(X[shuffled_idx], n_batches)
        y_batches_shuffled = np.array_split(y[shuffled_idx], n_batches)

        # Process each batch in the epoch
        for batch_idx in range(len(X_batches_shuffled)):
            X_batch = X_batches_shuffled[batch_idx]
            y_batch = y_batches_shuffled[batch_idx]
            
            # Create batch-specific objective and gradient functions
            def obj(w): return mse_loss(w, X_batch, y_batch)
            def grad(w): return mse_grad(w, X_batch, y_batch)
            
            # One step of SGD
            result = sgd(
                obj,
                params,
                grad,
                learning_rate=learning_rate,
                mass=0.9,
                maxiter=max_iter  # 1 if we're manually iterating through batches and epochs for sgd
            )
            
            # Update parameters
            params = result.x
            
            # Track loss and parameters
            current_loss = result.fun
            epoch_loss += current_loss
            all_losses.append(current_loss)
            all_params.append(params.copy())
        
        # Average loss for this epoch
        avg_epoch_loss = epoch_loss / len(X_batches_shuffled)
        epoch_losses.append(avg_epoch_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_epoch_loss:.6f}, Parameters: w={params[:-1]}, b={params[-1]:.4f}")
    
    return params, epoch_losses, all_losses, all_params

def plot_optimization_path(X, y, all_params, true_w, true_b):
    """
    Plot the optimization path on the objective function surface.
    Shows how the weights (w1, w2) move through the loss landscape.
    """
    # Extract w1 and w2 values from the optimization path
    w1_values = [p[0] for p in all_params]
    w2_values = [p[1] for p in all_params]
    
    # Create a grid of w1 and w2 values for plotting the loss surface
    w1_range = np.linspace(min(min(w1_values), 0) - 0.5, max(max(w1_values), true_w[0]) + 0.5, 100)
    w2_range = np.linspace(min(min(w2_values), true_w[1]) - 0.5, max(max(w2_values), 0) + 0.5, 100)
    w1_grid, w2_grid = np.meshgrid(w1_range, w2_range)
    
    # Calculate the loss for each point on the grid
    # Use the average bias from the optimization path for simplicity
    avg_bias = np.mean([p[2] for p in all_params])
    z_grid = np.zeros_like(w1_grid)
    
    for i in range(len(w1_range)):
        for j in range(len(w2_range)):
            w = np.array([w1_grid[j, i], w2_grid[j, i]])
            params = np.concatenate([w, [avg_bias]])
            z_grid[j, i] = mse_loss(params, X, y)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the loss surface
    surf = ax.plot_surface(w1_grid, w2_grid, z_grid, cmap='gist_gray', alpha=0.4, 
                          linewidth=0, antialiased=True)
    
    # Add contour plot at the bottom for better visualization
    contour = ax.contour(w1_grid, w2_grid, z_grid, levels=15, 
                         offset=z_grid.min(), cmap='viridis')
    
    # Downsample the optimization path for clarity (show every nth point)
    n = max(1, len(all_params) // 50)  # Show at most 50 points
    
    # Plot the optimization path with arrows to show direction
    for i in range(0, len(all_params)-n, n):
        w1_start, w2_start = w1_values[i], w2_values[i]
        w1_end, w2_end = w1_values[i+n], w2_values[i+n]
        
        # Get the loss values for the start and end points
        loss_start = mse_loss(all_params[i], X, y)
        loss_end = mse_loss(all_params[i+n], X, y)
        
        # Draw arrow from start to end point
        ax.quiver(w1_start, w2_start, loss_start, 
                 w1_end-w1_start, w2_end-w2_start, loss_end-loss_start,
                 color='red', arrow_length_ratio=0.1)
    
    # Mark the starting point
    ax.scatter(w1_values[0], w2_values[0], mse_loss(all_params[0], X, y), 
              color='blue', s=100, label='Start')
    
    # Mark the ending point
    ax.scatter(w1_values[-1], w2_values[-1], mse_loss(all_params[-1], X, y), 
              color='green', s=100, label='End')
    
    # Mark the true parameters point
    true_params = np.concatenate([true_w, [true_b]])
    true_loss = mse_loss(true_params, X, y)
    ax.scatter(true_w[0], true_w[1], true_loss, 
              color='gold', s=100, label='True parameters')
    
    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss value')
    
    # Add labels and title
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')
    ax.set_title('Optimization Path on Loss Surface')
    ax.legend()
    
    # Second plot: 2D contour with optimization path
    plt.figure(figsize=(10, 8))
    contour_plot = plt.contourf(w1_grid, w2_grid, z_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Loss value')
    
    # Plot the optimization path with arrows
    for i in range(0, len(all_params)-n, n):
        w1_start, w2_start = w1_values[i], w2_values[i]
        w1_end, w2_end = w1_values[i+n], w2_values[i+n]
        
        # Draw arrow from start to end point
        plt.arrow(w1_start, w2_start, w1_end-w1_start, w2_end-w2_start, 
                 color='red', head_width=0.05, head_length=0.1, length_includes_head=True)
    
    # Mark the starting point
    plt.scatter(w1_values[0], w2_values[0], color='blue', s=100, label='Start')
    
    # Mark the ending point
    plt.scatter(w1_values[-1], w2_values[-1], color='green', s=100, label='End')
    
    # Mark the true parameters point
    plt.scatter(true_w[0], true_w[1], color='gold', s=100, label='True parameters')
    
    plt.xlabel('Weight 1')
    plt.ylabel('Weight 2')
    plt.title('Contour Plot of Loss Function with Optimization Path')
    plt.legend()
    plt.tight_layout()

def main(n_epochs=1000, batch_size=50, max_iter=100, learning_rate=0.01):
    # Real model:
    # y = [2.5, -1.0] @ X + 0.5 + noise
    true_w = np.array([2.5, -1.0])  # True model parameters (weights)
    true_b = 0.5                     # True model bias

    # Generate training data
    n_samples = 1000
    X = np.random.randn(n_samples, 2)  # Features
    y = X @ true_w + true_b + np.random.randn(n_samples) * 0.5  # Target with noise

    # Split into batches with batch_size examples each
    n_batches = n_samples // batch_size

    # Run training for multiple epochs
    params, epoch_losses, all_losses, all_params = train_with_epochs(n_epochs, X, y, n_batches, learning_rate=learning_rate, max_iter=max_iter)

    # Gradient call count determines optimization complexity
    print(f"Function calls: {f_counter}\nGradient calls: {df_counter}")

    # Plot the learning curves
    plt.figure(figsize=(15, 5))

    # Plot 1: Loss per epoch
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs+1), epoch_losses, 'o-', linewidth=2)
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)

    # Plot 2: Loss per batch update
    plt.subplot(1, 2, 2)
    plt.plot(all_losses, linewidth=1.5)
    plt.title('Loss per Batch Update')
    plt.xlabel('Batch Updates')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot the optimization path on the loss surface
    plot_optimization_path(X, y, all_params, true_w, true_b)

    # Compare learned parameters to true parameters
    print("\nFinal Results:")
    print(f"True weights: {true_w}, bias: {true_b}")
    print(f"Learned weights: {params[:-1]}, bias: {params[-1]:.4f}")
    print(f"Final loss: {epoch_losses[-1]:.6f}")

    # Visualize parameter convergence
    plt.figure(figsize=(12, 5))

    # Plot weights convergence
    plt.subplot(1, 2, 1)
    w1_values = [p[0] for p in all_params]
    w2_values = [p[1] for p in all_params]
    plt.plot(w1_values, label='w1 (True: 2.5)')
    plt.plot(w2_values, label='w2 (True: -1.0)')
    plt.axhline(y=true_w[0], color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=true_w[1], color='g', linestyle='--', alpha=0.5)
    plt.title('Weight Convergence')
    plt.xlabel('Batch Updates')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)

    # Plot bias convergence
    plt.subplot(1, 2, 2)
    b_values = [p[-1] for p in all_params]
    plt.plot(b_values, label='b (True: 0.5)')
    plt.axhline(y=true_b, color='r', linestyle='--', alpha=0.5)
    plt.title('Bias Convergence')
    plt.xlabel('Batch Updates')
    plt.ylabel('Bias Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Sample size of training data is 1000 examples.
    # Batch size is 1000 for gradient descent, resulting in 1 batch.
    # Batch size is 50 for mini-batch gradient descent, for 20 batches.
    # Batch size is 1 for stochastic gradient descent, for 1000 batches.
    #
    # Gradient evaluations are set to 100.000 for the three methods.
    # This is given by n_epochs * n_batches * max_iter.
    #
    # After each batch, the parameters are updated. The updated model is used
    # for the next batch.
    #
    # Number of epochs is given so they all have the same number of gradient calls
    
    # Gradient Descent
    main(n_epochs=2500, batch_size=1000, max_iter=1) # TODO: test this

    # Mini-Batch Stochastic Gradient Descent
    #main(n_epochs=100, batch_size=50, max_iter=1) # TODO: then test this - faster convergence, check gradient call count

    # Stochastic Gradient Descent
    #main(n_epochs=5, batch_size=1, max_iter=1) # TODO: finally, test this - much faster convergence, check gradient call count
