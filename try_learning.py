import network_model.Network as N
import network_model.Layer as L
import network_model.GenerateData as GD
import numpy as np
import matplotlib.pyplot as plt

ds = GD.PatternDataset(num_samples=100, width=5, length=100, spike_prob=0.05, pattern_neurons=(0, 1, 2), num_patterns=1)

test_ds = GD.SequenceDataset(num_samples=100, width=10, delay=0, spike_threshold=0.05)

net = N.Network()

net.add(L.FCLayer(10, 10, is_first=True))
net.add(L.FCLayer(10, 10))
# print(net.layers[0].weights)

# net.layers[0].weights[:] = 0.5
# net.layers[1].weights[:] = 0.5

print(net.layers[0].weights)

# -------------------------
# TRAIN
# -------------------------
i = 0
for epoch in range(20):
    ds = GD.SequenceDataset(num_samples=100, width=10, delay=0, spike_threshold=0.1)
    for sample, target in ds:
        # print(f"Training sample: {i+1}")


        # print(sample.sum())
        # print(sample.shape)
        # exit()

        net.reset_state()

        for t in range(sample.shape[1]):
            input_spikes = sample[:, t]
            target_spikes = target[:, t]

            net(
                input_spikes,
                target_spikes,
                learn=True
            )

            # print(net.layers[0].weights)

        i += 1
    print(f"Epoch: {epoch}")
    print("Layer 1 Train Winner Count:")
    print(net.layers[0].winner_count)
    net.layers[0].winner_count = np.zeros(net.layers[0].layer_size)

    # print(net.layers[0].weights)
    # print(net.layers[0].post_trace)
    # print(net.layers[0].pre_trace)

    MSE = 0

    i = 0
    tp = 0
    fp = 0
    fn = 0
    for sample, target in test_ds:

        net.reset_state()

        sample_MSE = 0

        outputs = []

        for t in range(sample.shape[1]):
            input_spikes = sample[:, t]

            # NO TARGET HERE
            output = net(input_spikes)

            outputs.append(output.copy())

        outputs = np.array(outputs).T

        target_bool = target.astype(bool)
        output_bool = outputs.astype(bool)

        tp += np.logical_and(
            target_bool,
            output_bool
        ).sum()

        fp += np.logical_and(
            ~target_bool,
            output_bool
        ).sum()

        fn += np.logical_and(
            target_bool,
            ~output_bool
        ).sum()

        # sample_MSE = np.mean(np.square(outputs - target))

        # print(f"Sample {i+1}: {sample_MSE}")

        MSE += sample_MSE

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    MSE /= len(test_ds)

    print(f"Final MSE for Epoch {epoch+1}: {MSE}")
    print(f"Prec: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print("Layer 1:")
    print(np.round(net.layers[0].weights, 2))
    print("Layer 2:")
    print(np.round(net.layers[1].weights, 2))
    print("Layer 1 Test Winner Count:")
    print(net.layers[0].winner_count)
    net.layers[0].winner_count = np.zeros(net.layers[0].layer_size)

    # print(net.layers[0].weights)
    # print(net.layers[1].weights)

dt = np.array(
    net.layers[0].stdp_dt
)

dw = np.array(
    net.layers[0].stdp_dw
)

plt.scatter(
    dt,
    dw,
    alpha=0.1,
    s=5
)

plt.axhline(0)
plt.axvline(0)

plt.xlabel(
    r"$\Delta t = t_{post} - t_{pre}$"
)
plt.ylabel(
    r"$\Delta w$"
)

plt.title(
    "Empirical STDP Events During Training"
)

plt.show()