# Architectures

Train four small models from scratch. Each task has a held-out set and a reason for using its particular attention mask.

| Worked example | Experiment |
| --- | --- |
| [Bidirectional masked-token encoder](01%20Bidirectional%20masked-token%20encoder/README.md) | A sequence contains four symbols followed by a second copy of the four. After hiding one symbol, can a bidirectional encoder retrieve its partner? |
| [Causal decoder and next-token prediction](02%20Causal%20decoder%20and%20next-token%20prediction/README.md) | A three-symbol motif repeats three times. Its first occurrences are hard to guess, but every later token has an exact match in the prefix. |
| [Encoder-decoder sequence reversal](03%20Encoder-decoder%20sequence%20reversal/README.md) | Reverse a four-symbol input one symbol at a time. A small encoder reads the whole source, while the decoder attends to the encoded source and its own prefix. |
| [Vision Transformer on handwritten digits](04%20Vision%20Transformer%20on%20handwritten%20digits/README.md) | Cut each digit into sixteen patches and train a class token to classify the image. A confusion matrix shows which digits remain difficult at this resolution. |

The reversal and repeating-motif tasks measure generalisation to unseen symbol combinations. The image classifier uses stratified handwritten digits.
