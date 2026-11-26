"""Scaled dot-product and masked attention

Four library records contain a city or a year. A query retrieves a weighted average of the records it can see. One deliberately empty row makes the masking convention explicit.

Vaswani et al. (2017), Attention Is All You Need
https://arxiv.org/abs/1706.03762"""

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


def attention(q, k, v, allowed):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    active = allowed.any(axis=-1, keepdims=True)
    masked = np.where(allowed, scores, -np.inf)
    peak = np.where(active, masked.max(axis=-1, keepdims=True), 0.0)
    w = np.exp(masked - peak)
    total = w.sum(axis=-1, keepdims=True)
    w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    return w, w @ v


def main():
    q = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    k = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]])
    v = np.array([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0], [0.2, 0.8]])
    allowed = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool)
    w, out = attention(q, k, v, allowed)
    changed = v.copy()
    changed[0] += 100
    assert np.allclose(attention(q, k, changed, allowed)[1][1:], out[1:])
    assert np.allclose(w.sum(1), [1, 1, 0])
    report(
        ["Query", "Allowed keys", "Sum of weights", "City output", "Year output"],
        [
            (n, int(a.sum()), r.sum(), *o)
            for n, a, r, o in zip(["City", "Year", "Empty"], allowed, w, out)
        ],
    )

    fig, ax = plt.subplots(figsize=(6, 2.8), layout="constrained")
    im = ax.imshow(w, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for (i, j), value in np.ndenumerate(w):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center")
    ax.set(
        xticks=range(4),
        yticks=range(3),
        yticklabels=["City", "Year", "Empty"],
        xlabel="Record",
        ylabel="Query",
        title="Retrieval weights",
    )
    fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
