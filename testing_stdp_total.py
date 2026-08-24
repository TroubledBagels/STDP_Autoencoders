import numpy as np
from network_model.Layer import FCLayer

layer = FCLayer(1, 1, is_first=True)
layer.weights[:] = 0.5

sample = np.array([[1, 0, 0, 0, 0]])
target = np.array([[0, 0, 1, 0, 0]])

for t in range(sample.shape[1]):
    print(f"\n--- t={t} ---")
    print("input:", sample[:, t])
    print("target:", target[:, t])
    print("pre trace before:", layer.pre_trace.copy())
    print("post trace before:", layer.post_trace.copy())
    print("weight before:", layer.weights.copy())

    layer(
        sample[:, t],
        target=target[:, t], learn=True
    )

    print("pre trace after:", layer.pre_trace.copy())
    print("post trace after:", layer.post_trace.copy())
    print("weight after:", layer.weights.copy())