# Embeddings and Positions

Start with a six-symbol vocabulary. Then hold token content fixed and test what changes when positions move.

| Worked example | Experiment |
| --- | --- |
| [Token embeddings and tied output weights](01%20Token%20embeddings%20and%20tied%20output%20weights/README.md) | A language model can use one matrix both to look up tokens and to score the next token. Six symbols arranged on a ring give us a transition law we can check exactly. |
| [Sinusoidal positions and rotary embeddings](02%20Sinusoidal%20positions%20and%20rotary%20embeddings/README.md) | Why can an attention head recognise a separation of three tokens after both tokens move? Compare an absolute sinusoidal code with a rotation of a query and a key. |
| [ALiBi and attention distance](03%20ALiBi%20and%20attention%20distance/README.md) | Attention with linear biases gives each causal head a different cost for looking further back. A neutral content score isolates what that bias does. |

A shared shift leaves the rotary dot product unchanged to numerical precision. The ALiBi example quantifies the lag each head favours.
