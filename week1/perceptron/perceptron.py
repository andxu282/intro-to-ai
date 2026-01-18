import numpy as np
from dataset_generator import DatasetGenerator
from utils import Utils

MAX_NUM_EPOCHS = 1e6

class Perceptron:
    def __init__(self, num_dims):
        pass

    def predict(self, x):
        pass

    def train(self, dataset, num_epochs=None):
        pass

if __name__ == '__main__':
    dataset_generator = DatasetGenerator(num_points=20)
    dataset = dataset_generator.generate()
    perceptron = Perceptron(num_dims=2)
    (w, b), num_epochs = perceptron.train(dataset, num_epochs=100)
    print(f"Learned weights: {w}, bias: {b}, epochs: {num_epochs}")

    Utils.plot_decision_boundary(dataset, w, b)
