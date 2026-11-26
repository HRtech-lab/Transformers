# Transformers

Fifteen experiments follow the path from token embeddings through attention, encoder and decoder layers, and expert routing. The notebooks expose intermediate calculations so you can inspect what a query retrieves, how a causal cache changes decoding cost and which tokens reach each expert.

## Try a short example

The [short examples](examples/README.md) cover image patch tokenisation and root mean square normalisation (RMSNorm). The two scripts use NumPy. From the repository root, run

```sh
python examples/patch_tokens.py
```

## Course

| Section | Experiments |
| --- | --- |
| [Embeddings and Positions](Embeddings%20and%20Positions/README.md) | Token lookup and output weight tying, absolute sinusoids, rotary embeddings and ALiBi. |
| [Attention](Attention/README.md) | Masked, multi-head and cross-attention, causal caching, grouped and multi-query heads, online softmax, local and global attention. |
| [Architectures](Architectures/README.md) | Bidirectional masked-token prediction, causal language modelling, encoder-decoder transduction and a Vision Transformer. |
| [Mixture of Experts](Mixture%20of%20Experts/README.md) | Top-two expert dispatch, top-one Switch routing, auxiliary load loss and overflow. |

## Two results to inspect

At a prefix of 32 tokens, the single-layer cache stores 4,096 bytes of double-precision keys and values. It evaluates 528 query-key scores over the generation run. Dense recomputation at every prefix evaluates 11,440. Both paths produce the same vectors to floating-point precision. [Cache experiment](Attention/03%20Causal%20attention%20and%20the%20key-value%20cache/README.md)

![Cumulative attention scores during decoding](Attention/03%20Causal%20attention%20and%20the%20key-value%20cache/figure.svg)

The patch classifier reached 98.2% test accuracy on 450 handwritten digit images. Its confusion matrix shows which pairs it mixed up. [Vision Transformer experiment](Architectures/04%20Vision%20Transformer%20on%20handwritten%20digits/README.md)

## Run the examples

Use Python 3.12. From the repository root, install the listed dependencies and open a notebook, or run the matching Python script. Each notebook contains the recorded tables and figures.

```sh
python -m pip install -r requirements.txt
python "Attention/01 Scaled dot-product and masked attention/scaled_dot_product_and_masked_attention.py"
```

The sequence tasks and token grids are generated locally. Handwritten digit images are loaded from scikit-learn, using 8 × 8 images from the UCI optical digits test partition. Preprocessing scales intensities by 16 and the image-level split is stratified. Each chapter links its primary papers. [UCI dataset](https://doi.org/10.24432/C50P49) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [MIT code licence](LICENSE)
