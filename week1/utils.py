import numpy as np
from matplotlib import pyplot as plt

class Utils:
    @staticmethod
    def plot_decision_boundary(dataset, w, b):
        # Separate points by class for plotting
        green_points = [x for x, y in dataset if y == 1]
        red_points = [x for x, y in dataset if y == -1]
        
        # Plot dataset points
        if green_points:
            green_x = [p[0] for p in green_points]
            green_y = [p[1] for p in green_points]
            plt.scatter(green_x, green_y, c='green', label='y = 1', s=100)
        
        if red_points:
            red_x = [p[0] for p in red_points]
            red_y = [p[1] for p in red_points]
            plt.scatter(red_x, red_y, c='red', label='y = -1', s=100)
        
        # Plot the learned decision boundary: w[0]*x[0] + w[1]*x[1] + b = 0
        # Solving for x[1]: x[1] = -(w[0]*x[0] + b) / w[1]
        x_line = np.linspace(-1, 1, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        plt.plot(x_line, y_line, 'b-', linewidth=2, label=f'Learned boundary')
        
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.title('Perceptron Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.show()

    @staticmethod
    def plot_dataset(self, dataset):
        # Separate points by class
        green_points = [x for x, y in dataset if y == 1]
        red_points = [x for x, y in dataset if y == -1]
        # Plot
        if green_points:
            green_x = [p[0] for p in green_points]
            green_y = [p[1] for p in green_points]
            plt.scatter(green_x, green_y, c='green', label='y = 1', s=100)
        
        if red_points:
            red_x = [p[0] for p in red_points]
            red_y = [p[1] for p in red_points]
            plt.scatter(red_x, red_y, c='red', label='y = -1', s=100)
        
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.title('Dataset Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.show()