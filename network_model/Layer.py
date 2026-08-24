import numpy as np
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
    def __init__(self, input_num, layer_size, n_type=NeuronType.LIF_total, is_first=False):
        self.name = "FC"
        self.n_type = n_type
        self.input_num = input_num
        self.layer_size = layer_size
        self.u = np.zeros(layer_size)
        self.weights = np.random.uniform(
            0.02,
            0.15,
            size=(input_num, layer_size)
        )
        self.spike_record = []
        self.mem_pot_record = []
        self.post_trace = np.zeros(layer_size)
        self.A_plus = 0.0005
        self.A_minus = 0.00055
        self.is_first = is_first
        if self.is_first:
            self.pre_trace = np.zeros(input_num)

    def __call__(self, inputs, pre_trace=None, target=None, learn=False, competitive=False):

        if len(inputs) != self.input_num:
            raise ValueError(
                "The number of inputs is not equal to the number of neurons"
            )

        # Decay old post trace
        self.post_trace *= 0.9
        if self.is_first:
            self.pre_trace *= 0.9

        # Normal network forward pass
        new_mem_pot = np.dot(inputs, self.weights)
        spikes = self.update_u(new_mem_pot, competitive=competitive)

        # During training, use target spikes for STDP
        if target is not None:
            learning_spikes = target
        else:
            learning_spikes = spikes

        if learn:
            if self.is_first:
                self.update_weights(
                    self.pre_trace,
                    inputs,
                    learning_spikes,
                )
            elif pre_trace is not None:
                self.update_weights(
                    pre_trace,
                    inputs,
                    learning_spikes,
                )
            else:
                raise ValueError(
                    "Non-first layer requires presynaptic trace"
                )

        ret_trace = self.post_trace.copy()

        # For training, decide what should enter the post trace
        if target is not None:
            self.post_trace += target
        else:
            self.post_trace += spikes
        if self.is_first:
            self.pre_trace += inputs

        return spikes, ret_trace

    def update_u(self, mem_pot_update, competitive=False):
        if self.n_type == NeuronType.LIF_total and not competitive:
            self.u += mem_pot_update
            spikes = np.where(self.u >= 1, 1, 0)
            self.spike_record.append(spikes)
            self.u[spikes == 1] = 0
            self.u = np.clip(self.u, 0, None)
            self.u *= 0.9 # Decay membrane potential by 10%
            self.mem_pot_record.append(self.u.copy())
        elif self.n_type == NeuronType.LIF_total and competitive:
            self.u += mem_pot_update

            candidate_spikes = self.u >= 1

            if candidate_spikes.any():
                winner = np.argmax(
                    np.where(
                        candidate_spikes,
                        self.u,
                        -np.inf
                    )
                )

                spikes = np.zeros(
                    self.layer_size,
                    dtype=int
                )

                spikes[winner] = 1
            else:
                spikes = candidate_spikes.astype(int)

            self.spike_record.append(spikes.copy())

            self.u[candidate_spikes] = 0
            self.u *= 0.9
            self.mem_pot_record.append(self.u.copy())

        return spikes

    def update_weights(self, pre_trace, inputs, spikes):
        # # STDP update rule
        # print("UPDATE_WEIGHTS CALLED")
        #
        # ltp = self.A_plus * np.outer(pre_trace, spikes)
        # ltd = self.A_minus * np.outer(inputs, self.post_trace)
        #
        # print("A_plus:", self.A_plus)
        # print("A_minus:", self.A_minus)
        # print("pre_trace:", pre_trace)
        # print("inputs:", inputs)
        # print("spikes:", spikes)
        # print("post_trace:", self.post_trace)
        # print("LTP:", ltp)
        # print("LTD:", ltd)
        # print("weights BEFORE:", self.weights)
        #
        # self.weights += ltp
        # self.weights -= ltd
        #
        # print("weights BEFORE CLIP:", self.weights)
        #
        # self.weights = np.clip(self.weights, 0.0, 1.0)
        #
        # print("weights AFTER:", self.weights)
        ltp = (
            self.A_plus
            * np.outer(pre_trace, spikes)
            * (1.0 - self.weights)
        )

        ltd = (
            self.A_minus
            * np.outer(inputs, self.post_trace)
            * self.weights
        )

        # self.weights += self.A_plus * np.outer(pre_trace, spikes)
        # self.weights -= self.A_minus * np.outer(inputs, self.post_trace)
        self.weights += ltp - ltd
        self.weights = np.clip(self.weights, 0.0, 1.0)

    def reset_state(self):
        self.u[:] = 0
        self.post_trace[:] = 0

        if self.is_first:
            self.pre_trace[:] = 0

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