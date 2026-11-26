"""Apply root mean square normalisation across each token's feature dimension."""
import numpy as np


def rms_norm(tokens, gain, epsilon=1e-6):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rms = np.sqrt(np.mean(np.square(tokens), axis=-1, keepdims=True) + epsilon)
    return gain * (tokens / rms)


if __name__ == "__main__":
    tokens = np.array([[3., 4.], [0., 0.]])
    output = rms_norm(tokens, np.ones(2))
    assert np.isclose(np.mean(output[0] ** 2), 1) and np.all(output[1] == 0)
    print("Normalised tokens:")
    print(output.round(3))
