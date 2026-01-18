import numpy as np
from matplotlib import pyplot as plt

class DatasetGenerator:
    def __init__(self, num_points):
        self.num_points = num_points

    def generate(self):
        dataset = []
        for _ in range(self.num_points):
            x = np.random.uniform(-1, 1, size=2)
            y = np.sign(x[0] + x[1])
            dataset.append((x, y))
        return dataset
    


if __name__ == '__main__':
    dataset_generator = DatasetGenerator(num_points=20)
    