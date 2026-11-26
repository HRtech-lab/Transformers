# Mixture of Experts

Trace a token from its router logits to its expert output, then measure what happens when too many tokens choose the same expert.

| Worked example | Experiment |
| --- | --- |
| [Sparse top-two expert routing](01%20Sparse%20top-two%20expert%20routing/README.md) | Send a point to two of four small expert networks. Their gate probabilities determine how the two responses combine. |
| [Switch routing and expert capacity](02%20Switch%20routing%20and%20expert%20capacity/README.md) | An expert may receive more tokens than its buffer can hold. Track the token drops as routing demand grows less even. |

A gate probability, the number of assigned tokens and the number accepted under capacity are three distinct quantities.
