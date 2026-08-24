import numpy as np
import network_model.Network as N
import network_model.Layer as L

net = N.Network()

net.add(
    L.FCLayer(
        input_num=2,
        layer_size=1,
        is_first=True
    )
)

net.layers[0].weights[:] = 0.5

sample = np.array([
    [1, 0, 0, 0, 0],  # input A
    [0, 0, 0, 1, 0],  # input B
])

target = np.array([
    [0, 0, 1, 0, 0]
])

print("Initial weights:")
print(net.layers[0].weights)

for epoch in range(100):

    net.reset_state()

    for t in range(sample.shape[1]):

        input_spikes = sample[:, t]
        target_spikes = target[:, t]

        net(
            input_spikes,
            target_spikes,
            learn=True
        )

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}:",
            net.layers[0].weights.flatten()
        )

print("Final weights:")
print(net.layers[0].weights)