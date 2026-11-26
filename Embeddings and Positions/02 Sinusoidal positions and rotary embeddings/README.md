# Sinusoidal positions and rotary embeddings

Why can an attention head recognise a separation of three tokens after both tokens move? Compare an absolute sinusoidal code with a rotation of a query and a key.

Each pair of channels has its own angular frequency. The inner product of two rotated vectors depends on the difference between their positions. This experiment holds both content vectors fixed so that the position effect can be measured.

[Worked notebook](sinusoidal_positions_and_rotary_embeddings.ipynb) · [Python script](sinusoidal_positions_and_rotary_embeddings.py)

## Test relative position directly

Apply the same global shifts to both positions. Inner products should agree to floating-point precision.

## Compare the position maps

Left, colour encodes a coordinate of the original position embedding. Right, diagonals have the same query-key separation.

## Results from the recorded run

| Shared shift | Max dot-product error | Change in query norm |
| --- | --- | --- |
| 0 | 0.00e+00 | 0.00e+00 |
| 3 | 3.55e-15 | 0.00e+00 |
| 11 | 4.44e-15 | 0.00e+00 |
| 37 | 3.55e-15 | 0.00e+00 |

## Interpreting the result

The diagonals stay level because the vectors entering the rotation are fixed. With different token content, scores along a diagonal can vary.

![Sinusoidal positions and rotary embeddings, worked result](figure.svg)

## Exercise

Repeat the common shift test after adding absolute positions to the content vectors. Which terms in the expanded dot product change?

## References

[Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Su et al. (2021), RoFormer](https://arxiv.org/abs/2104.09864)
