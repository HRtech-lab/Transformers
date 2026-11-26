# Attention

Begin with a masked dot product and work towards caches, grouped queries, tiling and local windows. The numerical references make the mathematical assumptions testable.

| Worked example | Experiment |
| --- | --- |
| [Scaled dot-product and masked attention](01%20Scaled%20dot-product%20and%20masked%20attention/README.md) | Four library records contain a city or a year. A query retrieves a weighted average of the records it can see. One deliberately empty row makes the masking convention explicit. |
| [Multi-head self-attention and cross-attention](02%20Multi-head%20self-attention%20and%20cross-attention/README.md) | A bilingual glossary contains a source token and a word class. Two heads can consult the same memory for different attributes. |
| [Causal attention and the key-value cache](03%20Causal%20attention%20and%20the%20key-value%20cache/README.md) | During generation, a decoder can store keys and values for earlier tokens. The cached calculation should reproduce a full causal pass at every prefix. |
| [Multi-query and grouped-query attention](04%20Multi-query%20and%20grouped-query%20attention/README.md) | Many decoder query heads can share fewer key and value heads. Grouping reduces the retained keys and values during autoregressive decoding. |
| [Blockwise exact attention](05%20Blockwise%20exact%20attention/README.md) | A softmax can be accumulated without keeping a full matrix of attention scores. A running maximum is what makes the partial sums agree with dense attention. |
| [Sliding-window and global attention](06%20Sliding-window%20and%20global%20attention/README.md) | A local head can only read neighbouring tokens in one layer. Marking a single token global changes which parts of a long sequence can exchange information in two layers. |

The examples count score entries, bytes or reachable tokens. Hardware timings depend on kernels and devices.
