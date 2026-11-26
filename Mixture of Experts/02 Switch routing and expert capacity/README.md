# Switch routing and expert capacity

An expert may receive more tokens than its buffer can hold. Track the token drops as routing demand grows less even.

The Switch layer chooses one expert per token and scales its output by the selected probability from the full softmax. The auxiliary load loss uses the pre-capacity assignment fraction and the mean softmax probability of each expert.

[Worked notebook](switch_routing_and_expert_capacity.ipynb) · [Python script](switch_routing_and_expert_capacity.py)

## Measure demand, capacity and loss

Two groups have equal token counts. The skewed group receives an additive logit bias towards the first expert. Overflow tokens follow the identity residual.

## Compare demand with admitted tokens

At capacity factor 1.25, compare demand and completed expert calls. Token order resolves overflow here.

## Results from the recorded run

| Batch | Capacity factor | Slots per expert | Peak demand | Dropped tokens | Drop fraction | Auxiliary loss |
| --- | --- | --- | --- | --- | --- | --- |
| Balanced | 1.00 | 48 | 51 | 5 | 0.02604 | 1.00155 |
| Balanced | 1.25 | 60 | 51 | 0 | 0.00000 | 1.00155 |
| Balanced | 2.00 | 96 | 51 | 0 | 0.00000 | 1.00155 |
| Skewed | 1.00 | 48 | 165 | 117 | 0.60938 | 2.35297 |
| Skewed | 1.25 | 60 | 165 | 105 | 0.54688 | 2.35297 |
| Skewed | 2.00 | 96 | 165 | 69 | 0.35938 | 2.35297 |

## Interpreting the result

With 60 slots per expert, the balanced batch admitted all 192 tokens. In the skewed batch, 165 tokens selected one expert and 105 missed an expert update. Each of those tokens still followed the residual path.

The auxiliary term diagnoses the distribution of demand before clipping. Increasing capacity can lower the drop fraction even though that term is unchanged for the same logits.

![Switch routing and expert capacity, worked result](figure.svg)

## Exercise

Shuffle the skewed batch before dispatch and undo the shuffle afterwards. Check which individual tokens now receive an expert update.

## References

[Fedus, Zoph and Shazeer (2022), Switch Transformers](https://arxiv.org/abs/2101.03961)

[Shazeer et al. (2017), Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)
