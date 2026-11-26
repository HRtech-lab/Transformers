"""Multi-head self-attention and cross-attention

A bilingual glossary contains a source token and a word class. Two heads can consult the same memory for different attributes.

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


def softmax(logits, axis=-1):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def attend(queries, memory):
    n_heads = 2
    width = 2
    q = queries.reshape(len(queries), n_heads, width).transpose(1, 0, 2)
    k = memory.reshape(len(memory), n_heads, width).transpose(1, 0, 2)
    w = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(width))
    values = (w @ k).transpose(1, 0, 2).reshape(len(queries), -1)
    return w, values


def main():
    source = np.array(
        [[2.0, 0.0, 2.0, 0.0], [0.0, 2.0, 2.0, 0.0], [2.0, 0.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0]]
    )
    target = np.array([[2.0, 0.0, 0.0, 2.0], [0.0, 2.0, 2.0, 0.0], [1.0, 1.0, 0.0, 2.0]])
    self_w, _ = attend(source, source)
    cross_w, context = attend(target, source)
    rows = []
    for h in range(2):
        for i, weights in enumerate(cross_w[h]):
            rows.append(
                (h + 1, i + 1, np.argmax(weights) + 1, weights.max(), -weights @ np.log(weights))
            )
    assert context.shape == (3, 4) and np.allclose(cross_w.sum(-1), 1)
    report(
        ["Head", "Target query", "First maximum key", "Maximum weight", "Entropy (nats)"], rows
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.5), layout="constrained")
    for i, maps in enumerate([self_w, cross_w]):
        for j, w in enumerate(maps):
            ax = axes[i, j]
            im = ax.imshow(w, vmin=0, vmax=0.5, cmap="viridis", aspect="auto")
            ax.set(
                title=f"{['Self','Cross'][i]} · head {j+1}",
                xlabel="Source token",
                ylabel="Query",
                xticks=range(4),
                yticks=range(len(w)),
            )
    fig.colorbar(im, ax=axes, label="Weight", shrink=0.7)
    plt.show()


if __name__ == "__main__":
    main()
