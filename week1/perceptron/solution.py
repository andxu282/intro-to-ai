import numpy as np
from dataset_generator import DatasetGenerator
from utils import Utils

MAX_NUM_EPOCHS = 1e6

class Perceptron:
    def __init__(self, num_dims):
        self.w = np.random.rand(num_dims)
        self.b = np.random.random()

    def predict(self, x):
        return np.dot(self.w, x) + self.b

    def train(self, dataset, num_epochs=None):
        if num_epochs is None:
            num_epochs = MAX_NUM_EPOCHS

        for epoch in range(num_epochs):
            m = 0
            for (x, y) in dataset:
                if y * self.predict(x) < 0:
                    self.w = self.w + y * x
                    self.b = self.b + y
                    m += 1
            if m == 0:
                break
        
        return (self.w, self.b), epoch + 1

if __name__ == '__main__':
    dataset_generator = DatasetGenerator(num_points=20)
    dataset = dataset_generator.generate()
    perceptron = Perceptron(num_dims=2)
    (w, b), num_epochs = perceptron.train(dataset, num_epochs=100)
    print(f"Learned weights: {w}, bias: {b}, epochs: {num_epochs}")

    Utils.plot_decision_boundary(dataset, w, b)
