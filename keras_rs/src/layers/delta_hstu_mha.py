Python 3.12.5 (v3.12.5:ff3bc82f7c9, Aug  7 2024, 05:32:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
def get_valid_attn_mask_keras(
    causal: bool,
    N: int,
    seq_lengths,
    num_targets=None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
):
    ids = ops.reshape(ops.arange(0, N, dtype="int32"), (1, N))
    max_ids = ops.reshape(seq_lengths, (-1, 1, 1))

    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = ops.maximum(ids, 0)
        max_ids = max_ids - contextual_seq_len + 1

    if num_targets is not None:
        max_ids = max_ids - ops.reshape(num_targets, (-1, 1, 1))
        ids = ops.minimum(ids, max_ids)  # Simplified from torch.clamp
        row_ids = ops.broadcast_to(ops.reshape(ids, (-1, N, 1)), (ops.shape(seq_lengths)[0], N, N))
        col_ids = ops.broadcast_to(ops.reshape(ids, (-1, 1, N)), (ops.shape(seq_lengths)[0], N, N))
    else:
        row_ids = ops.broadcast_to(ops.reshape(ids, (N, 1)), (N, N))
        col_ids = ops.transpose(row_ids)
        row_ids = ops.reshape(row_ids, (1, N, N))
        col_ids = ops.reshape(col_ids, (1, N, N))
        max_ids = None

    row_col_dist = row_ids - col_ids
    valid_attn_mask = ops.reshape(ops.eye(N, dtype="bool"), (1, N, N))

    if not causal:
        row_col_dist = ops.where(row_col_dist > 0, row_col_dist, -row_col_dist)

    valid_attn_mask = ops.logical_or(valid_attn_mask, row_col_dist > 0)

    if max_attn_len > 0:
        if min_full_attn_seq_len > 0 and max_ids is not None:
            valid_attn_mask = ops.logical_and(
                valid_attn_mask,
                ops.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = ops.logical_and(valid_attn_mask, row_col_dist <= max_attn_len)

    if contextual_seq_len > 0 and max_ids is not None:
        valid_attn_mask = ops.logical_or(
            valid_attn_mask, ops.logical_and(row_ids == 0, col_ids < max_ids)
        )

    return valid_attn_mask

def delta_hstu_mha_keras(
    max_seq_len: int,
    alpha: float,
    delta_q: keras.KerasTensor,
    k: keras.KerasTensor,
    v: keras.KerasTensor,
    seq_offsets: keras.KerasTensor,
    num_targets: Optional[keras.KerasTensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    enable_tma: bool = False,
) -> keras.KerasTensor:
    """
    Keras 3 implementation of delta_hstu_mha using only keras.ops.
    """
    L, H, D = ops.shape(delta_q)
    B = ops.shape(seq_offsets)[0] - 1
    DeltaSize = L // B

    # Assertions (converted to ops.assert_true)
    ops.assert_true(max_seq_len > 0, "max_seq_len must be larger than 0")
    ops.assert_true(ops.ndim(delta_q) == 3, "delta_q must be 3-D")
    ops.assert_true(L % B == 0, "delta_q must be padded")
    ops.assert_true(ops.ndim(k) == 3, "k must be 3-D")
    ops.assert_true(ops.shape(k)[1] == H, "wrong k shape[1]")
    ops.assert_true(ops.shape(k)[2] == D, "wrong k shape[2]")
    ops.assert_true(ops.ndim(v) == 3, "v must be 3-D")
    ops.assert_true(ops.shape(v)[1] == H, "wrong v shape[1]")

    # Convert delta_q to expected shape (B, H, L//B, D) and transpose
    delta_q = ops.reshape(delta_q, (B, -1, H, D))
    delta_q = ops.transpose(delta_q, perm=[0, 2, 1, 3]) # (B, H, L//B, D)

    # Jagged to padded dense conversion for K and V
    # Reshape k and v to be flattened before jagged_to_padded_dense
    k_flat = ops.reshape(k, (-1, H * D))
    v_flat = ops.reshape(v, (-1, ops.shape(v)[2])) # Use V from v.shape directly

    full_k_padded = jagged_to_padded_dense(
        values=k_flat,
        offsets=seq_offsets,
        max_lengths=[max_seq_len], # max_seq_len should be a list/tuple for the function
        padding_value=0.0,
    )
    full_k = ops.reshape(full_k_padded, (B, -1, H, D))
    full_k = ops.transpose(full_k, perm=[0, 2, 1, 3]) # (B, H, max_seq_len, D)

    full_v_padded = jagged_to_padded_dense(
        values=v_flat,
        offsets=seq_offsets,
        max_lengths=[max_seq_len],
        padding_value=0.0,
    )
    full_v = ops.reshape(full_v_padded, (B, -1, ops.shape(v)[2]))
    full_v = ops.transpose(full_v, perm=[0, 2, 1, 3]) # (B, H, max_seq_len, V)

    # Scaled dot-product attention scores
    # qk_attn = delta_q @ full_k.transpose(-2, -1) * alpha
    # Using einsum for clarity, similar to PyTorch example
    qk_attn = ops.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha

    # Apply SiLU activation and scaling
    qk_attn = ops.silu(qk_attn) / ops.cast(max_seq_len, qk_attn.dtype)

    # Get the attention mask
    seq_lengths_tensor = seq_offsets[1:] - seq_offsets[:-1]
    full_valid_attn_mask = _get_valid_attn_mask_keras(
        device=None, # Keras ops don't require device argument
        causal=True,
        N=max_seq_len,
        seq_lengths=seq_lengths_tensor,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )


    mask_indices = ops.arange(max_seq_len, dtype="int32")
    mask_indices = ops.reshape(mask_indices, (1, -1)) # (1, max_seq_len)

    # Ensure seq_lengths is broadcastable
    if ops.ndim(seq_lengths_tensor) == 1:
        seq_lengths_tensor_reshaped = ops.expand_dims(seq_lengths_tensor, axis=-1) # (B, 1)
    else:
        seq_lengths_tensor_reshaped = seq_lengths_tensor # Assume it's already (B, 1) or similar

    # Condition for valid attention within the actual sequence length
    row_in_seq_mask = ops.logical_and(
        mask_indices >= (seq_lengths_tensor_reshaped - DeltaSize),
        mask_indices < seq_lengths_tensor_reshaped
    )
    if num_targets is None and ops.shape(full_valid_attn_mask)[0] == 1:
         full_valid_attn_mask = ops.broadcast_to(full_valid_attn_mask, (B, max_seq_len, max_seq_len))


    query_valid_indices = ops.arange(DeltaSize, device=delta_q.device) # (DeltaSize,)

    # Let's try to construct this mask:
    batch_indices = ops.arange(B, device=delta_q.device)
    query_indices = ops.arange(DeltaSize, device=delta_q.device)
    key_indices = ops.arange(max_seq_len, device=delta_q.device)

    # Meshgrid for batch, query, and key indices to map with full_valid_attn_mask
    b_mesh, q_mesh = ops.meshgrid(batch_indices, query_indices, indexing='ij')
    k_mesh = key_indices # This will be broadcast

    flat_mask_indices = ops.flatten(row_in_seq_mask) # (B * max_seq_len,)
    flat_full_attn_mask = ops.flatten(full_valid_attn_mask, start_axis=0, stop_axis=1) # (B * max_seq_len, max_seq_len)


    query_indices_b = ops.repeat(ops.arange(DeltaSize, device=delta_q.device), B) # (B * DeltaSize)
    batch_indices_for_indexing = ops.repeat(ops.arange(B, device=delta_q.device), DeltaSize) # (B * DeltaSize)



    # Ensure `full_valid_attn_mask` has `B` batches.
    if ops.shape(full_valid_attn_mask)[0] == 1 and B > 1:
        full_valid_attn_mask = ops.broadcast_to(full_valid_attn_mask, (B, max_seq_len, max_seq_len))


    # If `full_valid_attn_mask` is (B, max_seq_len, max_seq_len), slice it.
    if ops.shape(full_valid_attn_mask)[1] == max_seq_len and ops.shape(full_valid_attn_mask)[2] == max_seq_len:
        # Ensure we have DeltaSize rows if available
        if ops.shape(full_valid_attn_mask)[1] >= DeltaSize:
            masked_attn_scores = qk_attn * ops.expand_dims(full_valid_attn_mask[:, :DeltaSize, :], axis=1)
        else:

...             masked_attn_scores = qk_attn * ops.expand_dims(full_valid_attn_mask, axis=1)
...     else:
... 
...         masked_attn_scores = qk_attn * ops.expand_dims(full_valid_attn_mask, axis=1)
... 
... 
...     query_idx_tensor = ops.arange(DeltaSize, dtype="int32")
... 
... 
...     row_in_seq_mask_pt_style = ops.logical_and(
...         ops.expand_dims(key_indices, axis=0) >= (ops.expand_dims(seq_lengths_tensor, axis=-1) - DeltaSize),
...         ops.expand_dims(key_indices, axis=0) < ops.expand_dims(seq_lengths_tensor, axis=-1)
...     )
...     query_indices_for_mask = ops.arange(DeltaSize, dtype="int32")
... 
... 
...     if ops.shape(full_valid_attn_mask)[0] == 1 and B > 1:
...         full_valid_attn_mask = ops.broadcast_to(full_valid_attn_mask, (B, max_seq_len, max_seq_len))
... 
... 
...     valid_attn_mask_for_qk = ops.gather(full_valid_attn_mask, query_idx_tensor, axis=1) # (B, DeltaSize, max_seq_len)
... 
...     qk_attn_masked = qk_attn * ops.expand_dims(valid_attn_mask_for_qk, axis=1)
... 
... 
...     attn_output = ops.einsum("bhxd,bhyd->bhxv", qk_attn_masked, full_v)
... 
...     # Transpose and reshape the output
...     attn_output = ops.transpose(attn_output, perm=[0, 2, 1, 3]) # (B, max_seq_len, H, V)
...     final_output = ops.reshape(attn_output, (-1, H, ops.shape(v)[2])) # (L, H, V)
... 
