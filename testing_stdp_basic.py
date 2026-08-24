import network_model.Layer as L
import numpy as np

layer = L.FCLayer(1, 1)

layer.weights[:] = 0.5

# Pretend pre neuron fired recently
pre_trace = np.array([0.8])

# No current pre spike
pre_spikes = np.array([0])

# Post spikes now
post_spikes = np.array([1])

# Old post trace is zero
layer.trace[:] = 0

print("Before:", layer.weights.copy())

layer.update_weights(
    pre_trace,
    pre_spikes,
    post_spikes
)

print("After:", layer.weights)

layer.weights[:] = 0.5

# No recent pre spike
pre_trace = np.array([0.0])

# Pre spikes now
pre_spikes = np.array([1])

# Post fired recently
layer.trace[:] = 0.8

# Post does not spike now
post_spikes = np.array([0])

print("Before:", layer.weights.copy())

layer.update_weights(
    pre_trace,
    pre_spikes,
    post_spikes
)

print("After:", layer.weights)