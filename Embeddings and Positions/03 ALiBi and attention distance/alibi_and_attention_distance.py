"""ALiBi and attention distance

Attention with linear biases gives each causal head a different cost for looking further back. A neutral content score isolates what that bias does.

Press, Smith and Lewis (2022), Train Short, Test Long
https://arxiv.org/abs/2108.12409"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def report(columns, rows):
    frame = pd.DataFrame(rows, columns=columns)
    try:
        from IPython import get_ipython

        interactive = get_ipython() is not None
    except ImportError:
        interactive = False
    if interactive:
        from IPython.display import display

        display(frame.round(5))
    else:
        print(frame.round(5).to_string(index=False))


def softmax(logits, axis=-1):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def main():
    heads = 4
    slopes = 2.0 ** (-8 * np.arange(1, heads + 1) / heads)
    rows = []
    for length in [12, 32]:
        distances = np.arange(length)
        weights = softmax(-slopes[:, None] * distances)
        for i, (slope, w) in enumerate(zip(slopes, weights)):
            rows.append((length, i + 1, slope, distances @ w, w[:4].sum()))
    report(["Length", "Head", "Slope", "Expected distance", "Mass on nearest four"], rows)
    assert np.allclose(weights.sum(1), 1)

    fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    im = ax.imshow(weights, aspect="auto", cmap="magma")
    ax.set(
        xlabel="Distance into the past",
        ylabel="Head",
        title="ALiBi at a sequence length of 32",
        yticks=range(heads),
        yticklabels=range(1, heads + 1),
    )
    fig.colorbar(im, ax=ax, label="Attention probability")
    plt.show()


if __name__ == "__main__":
    main()
