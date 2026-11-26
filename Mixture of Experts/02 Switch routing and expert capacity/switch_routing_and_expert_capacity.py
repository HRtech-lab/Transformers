"""Switch routing and expert capacity

An expert may receive more tokens than its buffer can hold. Track the token drops as routing demand grows less even.

Fedus, Zoph and Shazeer (2022), Switch Transformers
https://arxiv.org/abs/2101.03961
Shazeer et al. (2017), Outrageously Large Neural Networks
https://arxiv.org/abs/1701.06538"""

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


import torch
import torch.nn.functional as F
from torch import nn


def seed():
    torch.manual_seed(17)
    torch.set_num_threads(1)


def route(logits, x, scale, capacity_factor):
    p = logits.softmax(-1)
    top = p.argmax(1)
    n, experts = p.shape
    cap = int(np.ceil(capacity_factor * n / experts))
    loads = torch.bincount(top, minlength=experts)
    kept = torch.zeros_like(loads)
    contribution = torch.zeros_like(x)
    for expert in range(experts):
        offered = torch.where(top == expert)[0]
        accepted = offered[:cap]
        kept[expert] = len(accepted)
        contribution[accepted] = p[accepted, expert] * scale[expert] * x[accepted]
    freq = loads.float() / n
    aux = experts * (freq * p.mean(0)).sum()
    return x + contribution, loads, kept, aux, cap


def main():
    seed()
    n = 192
    scores = torch.randn(n, 4)
    values = torch.linspace(0.1, 1, n)
    scale = torch.tensor([0.3, 1.2, -0.1, 0.6])
    rows = []
    cases = {}
    for name, bias in [("Balanced", [0.0, 0.0, 0.0, 0.0]), ("Skewed", [2.2, 0.0, 0.0, 0.0])]:
        logits = (scores + torch.tensor(bias)).requires_grad_()
        for factor in (1.0, 1.25, 2.0):
            output, loads, kept, aux, cap = route(logits, values, scale, factor)
            missed = int((loads - kept).sum())
            assert int(kept.sum()) + missed == n and bool((kept <= cap).all())
            assert int(torch.isclose(output, values).sum()) == missed
            rows.append(
                (name, factor, cap, int(loads.max()), missed, missed / n, float(aux.detach()))
            )
            if factor == 1.25:
                cases[name] = (loads.numpy(), kept.numpy())
        aux.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
    report(
        [
            "Batch",
            "Capacity factor",
            "Slots per expert",
            "Peak demand",
            "Dropped tokens",
            "Drop fraction",
            "Auxiliary loss",
        ],
        rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), layout="constrained", sharey=True)
    for ax, (name, (offer, keep)) in zip(axes, cases.items()):
        index = np.arange(4)
        ax.bar(index - 0.17, offer, 0.34, label="Offered", color="#8eacbb")
        ax.bar(index + 0.17, keep, 0.34, label="Admitted", color="#325f57")
        ax.set(xticks=index, xticklabels=index + 1, xlabel="Expert", title=name + " token batch")
    axes[0].set_ylabel("Tokens")
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
