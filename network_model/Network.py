import Layer as L
import Neurons as N
import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def __str__(self):
        string = ""
        for i, layer in enumerate(self.layers):
            string += f"{i+1}: " + str(layer) + "\n"

        return string

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def get_spike_records(self):
        spike_records = []
        for layer in self.layers:
            spike_records.append(layer.spike_record)

        return spike_records

    def get_mem_pot_records(self):
        mem_pot_records = []
        for layer in self.layers:
            mem_pot_records.append(layer.mem_pot_record)

        return mem_pot_records

def interpret_records(spike_records: list):
    i = 1
    for record in spike_records:
        print(f"Layer {i}:")
        for t in record:
            for u in t:
                if u % 1 == 0:
                    print(f"{u} ", end='')
                else:
                    print(f"{u:.2f} ", end='')
            print()
        i += 1

if __name__ == "__main__":
    network = Network()
    network.add(L.FCLayer(2, 4))
    network.add(L.FCLayer(4, 6))

    test_inp = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])

    print(network)

    for inp in test_inp:
        network(inp)

    interpret_records(network.get_spike_records())

    interpret_records(network.get_mem_pot_records())