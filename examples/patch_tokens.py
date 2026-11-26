"""Convert a single-channel image to ordered, non-overlapping patch tokens."""
import numpy as np


def patch_tokens(image, size):
    height, width = image.shape
    if size <= 0 or height % size or width % size:
        raise ValueError("Patch size must be positive and divide both dimensions")
    grid = image.reshape(height // size, size, width // size, size)
    # Group the two patch-grid axes before flattening each patch.
    return grid.transpose(0, 2, 1, 3).reshape(-1, size * size)


if __name__ == "__main__":
    tokens = patch_tokens(np.arange(16).reshape(4, 4), 2)
    assert tokens.tolist() == [[0, 1, 4, 5], [2, 3, 6, 7],
                               [8, 9, 12, 13], [10, 11, 14, 15]]
    print("One row per image patch:")
    print(tokens)
