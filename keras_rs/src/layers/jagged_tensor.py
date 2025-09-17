import keras
from keras import ops
from typing import List, Optional, Tuple

# --- Core Jagged Primitive ---

def keras_jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
    """
    Keras 3 implementation to convert jagged tensor (values) into a padded dense tensor.
    Required by both split and concat operations.
    """
    offsets = offsets[0] if isinstance(offsets, list) else offsets
    B = ops.shape(offsets)[0] - 1
    max_len = max_lengths[0]
    D_flat = ops.shape(values)[-1]
    if ops.shape(values)[0] == 0:
        return ops.full((B, max_len, D_flat), padding_value, dtype=values.dtype)

    def pad_one(i):
        start = offsets[i]; end = offsets[i+1]
        seq_len = end - start 
        seq = ops.slice(values, [start, 0], [seq_len, D_flat])
        if ops.equal(seq_len, 0):
             return ops.full((max_len, D_flat), padding_value, dtype=values.dtype)
        if seq_len < max_len:
            padding_shape = ops.stack([max_len - seq_len, D_flat])
            padding = ops.full(padding_shape, padding_value, dtype=values.dtype)
            return ops.concatenate([seq, padding], axis=0)
        else:
            return seq[:max_len]

    idxs = ops.arange(B, dtype='int32')
    return ops.map(pad_one, idxs)

def keras_dense_to_jagged(
    dense: keras.KerasTensor,
    x_offsets: List[keras.KerasTensor],
) -> keras.KerasTensor:
    """Keras 3 implementation to convert a padded dense tensor back into a jagged tensor."""
    seq_offsets = x_offsets[0]
    B = ops.shape(seq_offsets)[0] - 1
    N = ops.shape(dense)[1] 
    D_flat = ops.shape(dense)[2] 
    token_range = ops.arange(N)
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    mask = ops.expand_dims(token_range, axis=0) < ops.expand_dims(seq_lengths, axis=1)
    
    flattened = ops.reshape(dense, [-1, D_flat])
    flattened_mask = ops.reshape(mask, [-1])

    return flattened[flattened_mask]

# --- SPLIT LOGIC ---

def keras_split_2D_jagged_jagged(
    max_seq_len: int, values: keras.KerasTensor, offsets_left: keras.KerasTensor, offsets_right: keras.KerasTensor,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Core splitting kernel logic."""
    D_flat = ops.shape(values)[1]
    offsets = offsets_left + offsets_right
    padded_values_bnd = keras_jagged_to_padded_dense(values=values, offsets=[offsets], max_lengths=[max_seq_len], padding_value=0.0)
    padded_values = ops.reshape(padded_values_bnd, [-1, D_flat])
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    mask = ops.reshape(ops.arange(max_seq_len, dtype='int32'), [1, -1])
    lengths_left_broadcast = ops.reshape(lengths_left, [-1, 1])
    lengths_right_combined = ops.reshape(lengths_left + lengths_right, [-1, 1])
    mask_left = mask < lengths_left_broadcast
    mask_right = ops.logical_and(mask >= lengths_left_broadcast, mask < lengths_right_combined)
    
    return padded_values[ops.reshape(mask_left, [-1])], padded_values[ops.reshape(mask_right, [-1])]

def keras_split_2D_jagged_resolver(
    max_seq_len: int, values: keras.KerasTensor, max_len_left: Optional[int], max_len_right: Optional[int], offsets_left: Optional[keras.KerasTensor], offsets_right: Optional[keras.KerasTensor],
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Resolves optional offsets for split operation."""
    L_total = ops.shape(values)[0]
    offsets_left_non_optional = offsets_left
    if offsets_left is None:
        B = L_total // max_len_left
        offsets_left_non_optional = max_len_left * ops.arange(B + 1, dtype='int32')
    offsets_right_non_optional = offsets_right
    if offsets_right is None:
        B = L_total // max_len_right
        offsets_right_non_optional = max_len_right * ops.arange(B + 1, dtype='int32')
    
    return keras_split_2D_jagged_jagged(max_seq_len=max_seq_len, values=values, offsets_left=offsets_left_non_optional, offsets_right=offsets_right_non_optional)

def split_2D_jagged(
    max_seq_len: int, values: keras.KerasTensor, total_len_left: Optional[int] = None, total_len_right: Optional[int] = None, max_len_left: Optional[int] = None, max_len_right: Optional[int] = None, offsets_left: Optional[keras.KerasTensor] = None, offsets_right: Optional[keras.KerasTensor] = None, kernel=None,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Top-level wrapper for splitting 2D jagged tensors."""
    assert ops.ndim(values) == 2, "values must be 2D"
    assert offsets_left is not None or offsets_right is not None, "offsets_left and offsets_right cannot be None at the same time"
    if offsets_left is None: assert max_len_left is not None, "max_len_left must be provided when offsets_left is None"
    if offsets_right is None: assert max_len_right is not None, "max_len_right must be provided when offsets_right is None"
    if offsets_left is not None and offsets_right is not None: ops.assert_equal(ops.shape(offsets_left)[0], ops.shape(offsets_right)[0])
    return keras_split_2D_jagged_resolver(max_seq_len=max_seq_len, values=values, max_len_left=max_len_left, max_len_right=max_len_right, offsets_left=offsets_left, offsets_right=offsets_right)


# --- CONCAT LOGIC ---

def keras_concat_2D_jagged_jagged(
    values_left: keras.KerasTensor, values_right: keras.KerasTensor, max_len_left: int, max_len_right: int, offsets_left: keras.KerasTensor, offsets_right: keras.KerasTensor,
) -> keras.KerasTensor:
    """Core concatenation kernel logic (sequence concatenation)."""
    max_seq_len = max_len_left + max_len_right
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    padded_left = keras_jagged_to_padded_dense(values=values_left, offsets=[offsets_left], max_lengths=[max_len_left], padding_value=0.0) 
    padded_right = keras_jagged_to_padded_dense(values=values_right, offsets=[offsets_right], max_lengths=[max_len_right], padding_value=0.0) 
    concatted_dense = ops.concatenate([padded_left, padded_right], axis=1) 
    mask = ops.reshape(ops.arange(max_seq_len, dtype='int32'), [1, -1])
    lengths_left_broadcast = ops.reshape(lengths_left, [-1, 1])
    lengths_right_broadcast = ops.reshape(lengths_right, [-1, 1])
    mask = ops.logical_or(mask < lengths_left_broadcast, ops.logical_and(mask >= max_len_left, mask < max_len_left + lengths_right_broadcast))
    return concatted_dense[ops.reshape(mask, [-1])]


def pytorch_concat_2D_jagged_resolver(
    values_left: keras.KerasTensor, values_right: keras.KerasTensor, max_len_left: Optional[int], max_len_right: Optional[int], offsets_left: Optional[keras.KerasTensor], offsets_right: Optional[keras.KerasTensor],
) -> keras.KerasTensor:
    """Resolves optional offsets for concat operation."""
    L_total = ops.shape(values_left)[0]
    offsets_left_non_optional = offsets_left
    if offsets_left is None:
        B = L_total // max_len_left
        offsets_left_non_optional = max_len_left * ops.arange(B + 1, dtype='int32')
    offsets_right_non_optional = offsets_right
    if offsets_right is None:
        B = L_total // max_len_right
        offsets_right_non_optional = max_len_right * ops.arange(B + 1, dtype='int32')
    
    # Calculate final max_len (required by the core kernel)
    if max_len_left is None: max_len_left_final = ops.max(offsets_left_non_optional[1:] - offsets_left_non_optional[:-1])
    else: max_len_left_final = max_len_left
    if max_len_right is None: max_len_right_final = ops.max(offsets_right_non_optional[1:] - offsets_right_non_optional[:-1])
    else: max_len_right_final = max_len_right
        
    return keras_concat_2D_jagged_jagged(
        values_left=values_left, values_right=values_right, max_len_left=max_len_left_final, max_len_right=max_len_right_final,
        offsets_left=offsets_left_non_optional, offsets_right=offsets_right_non_optional,
    )

def concat_2D_jagged(
    max_seq_len: int, values_left: keras.KerasTensor, values_right: keras.KerasTensor, max_len_left: Optional[int] = None, max_len_right: Optional[int] = None, offsets_left: Optional[keras.KerasTensor] = None, offsets_right: Optional[keras.KerasTensor] = None, kernel=None,
) -> keras.KerasTensor:
    """Top-level wrapper for concatenating 2D jagged tensors."""
    assert ops.ndim(values_left) == 2, "values_left must be 2D"
    assert ops.ndim(values_right) == 2, "values_right must be 2D"
    ops.assert_equal(ops.shape(values_left)[1], ops.shape(values_right)[1])
    
    return pytorch_concat_2D_jagged_resolver(
        values_left=values_left, values_right=values_right, max_len_left=max_len_left, max_len_right=max_len_right,
        offsets_left=offsets_left, offsets_right=offsets_right,
    )
