import numpy as np
import Neurons as N
from enum import Enum

'''
Notes on the fully-connected layer:
 - Need to define the size of the layer, the number of inputs (to define weight matrix)
'''

class NeuronType(Enum):
    LIF_total = 1
    LIF_subtract = 2
    Hodgkin = 3

class FCLayer:
    def __init__(self, input_num, layer_size, n_type=NeuronType.LIF_total):
        self.name = "FC"
        self.n_type = n_type
        self.input_num = input_num
        self.layer_size = layer_size
        self.u = np.zeros(layer_size)
        self.weights = np.random.normal(size=(input_num, layer_size), loc=0.5, scale=0.5)
        self.spike_record = []
        self.mem_pot_record = []

    def __call__(self, inputs):
        # Inputs: (1, input_num)
        # Weights: (input_num, layer_size)
        # Output: (1, layer_size)

        if len(inputs) != self.input_num:
            raise ValueError("The number of inputs is not equal to the number of neurons")

        new_mem_pot = np.dot(inputs, self.weights)
        spikes = self.update_u(new_mem_pot)
        # print(self.u)

        return spikes

    def update_u(self, mem_pot_update):
        if self.n_type == NeuronType.LIF_total:
            self.u += mem_pot_update
            spikes = np.where(self.u > 1, 1, 0)
            self.spike_record.append(spikes)
            self.u -= 1000 * spikes
            self.u = np.clip(self.u, 0, None)
            self.u *= 0.9 # Decay membrane potential by 10%
            print(self.u)
            self.mem_pot_record.append(self.u.copy())

        return spikes

    def __str__(self):
        return f"{self.name}: {self.input_num} in -> {self.layer_size} out"

    def __repr__(self):
        return f"{self.name}: {self.input_num} in -> {self.layer_size} out"


if __name__ == "__main__":
    layer1 = FCLayer(2, 4,)
    print(layer1)

    print(layer1.weights)

    test_inp = np.array([1, 1])
    print(test_inp)

    print(layer1(test_inp))