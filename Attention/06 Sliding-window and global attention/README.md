# Sliding-window and global attention

A local head can only read neighbouring tokens in one layer. Marking a single token global changes which parts of a long sequence can exchange information in two layers.

Build the bidirectional local and global connection pattern used by Longformer. An explicit sparse calculation visits just the allowed keys. Both calculations here share projections for all tokens to isolate connectivity.

[Worked notebook](sliding_window_and_global_attention.ipynb) · [Python script](sliding_window_and_global_attention.py)

## Verify the output on a small sequence

The dense mask is built for visualisation and verification. A second, independently constructed dense reference checks both the chosen neighbours and their attention outputs.

## Inspect the information paths

A marked cell means a path exists under the mask. The amount of information a trained head sends along that path depends on its learned weights.

## Results from the recorded run

| Pattern | Active scores | All dense scores | Direct keys at 15 | Reach after 2 layers | Max abs error |
| --- | --- | --- | --- | --- | --- |
| Window | 149 | 961 | 5 | 9 | 3.33e-16 |
| Window + global | 205 | 961 | 6 | 31 | 3.33e-16 |

## Interpreting the result

A global token can read every input at layer one. A local token that reads it at layer two can therefore receive information from the full sequence. The visualisation constructs dense matrices only to verify this small case.

![Sliding-window and global attention, worked result](figure.svg)

## Exercise

Move the global token from the beginning to the middle and compare the number of active scores with the two-layer reach.

## References

[Beltagy, Peters and Cohan (2020), Longformer](https://arxiv.org/abs/2004.05150)

[Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762)
