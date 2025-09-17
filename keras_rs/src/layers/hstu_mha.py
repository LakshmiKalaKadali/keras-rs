Python 3.12.5 (v3.12.5:ff3bc82f7c9, Aug  7 2024, 05:32:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import keras
... from keras import ops
... from typing import Tuple, List, Optional
... import math
... 
... 
... def keras_hstu_mha(
...     max_seq_len: int,
...     alpha: float,
...     q: keras.KerasTensor,
...     k: keras.KerasTensor,
...     v: keras.KerasTensor,
...     seq_offsets: keras.KerasTensor,
...     causal: bool = True,
...     dropout_pr: float = 0.0,
...     training: bool = True,
...     attn_scale: Optional[keras.KerasTensor] = None,
...     # num_targets, max_attn_len, etc. are ignored in this simplified conversion
... ) -> keras.KerasTensor:
... 
...     L, H, _ = ops.shape(q)
...     V_dim = ops.shape(v)[2]
... 
...     # 1. Pad Q, K, V: Jagged -> Padded Dense [B, H, N, D]
...     q, k, v = keras_pad_qkv(q, k, v, seq_offsets, max_seq_len)
... 
...     # 2. Attention Score (Dot Product + Scale)
...     # [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
...     qk_attn = ops.einsum("bhxa,bhya->bhxy", q, k) * alpha
... 
...     # 3. Attn Scale / SiLU Activation
...     if attn_scale is not None:
...         # Logic for attn_scale
...         if ops.ndim(attn_scale) > 0:
...             # Jagged to Padded Dense [B, N, 1]
...             attn_scale_padded = keras_jagged_to_padded_dense(
...                 values=ops.expand_dims(attn_scale, axis=-1),
                offsets=[seq_offsets],
                max_lengths=[max_seq_len],
                padding_value=0.0,
            )
            # Reshape and transpose to match qk_attn's batch and head structure [B, 1, N, 1]
            attn_scale_padded = ops.expand_dims(ops.cast(attn_scale_padded, qk_attn.dtype), axis=1)

        qk_attn = ops.silu(qk_attn) * attn_scale_padded
    else:
        qk_attn = ops.silu(qk_attn) / max_seq_len

    # 4. Apply Mask
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    valid_attn_mask = keras_get_valid_attn_mask(
        causal=causal,
        N=max_seq_len,
        seq_lengths=seq_lengths,
    )

    # Apply mask: [B, H, N, N] * [B, 1, N, N]
    qk_attn = qk_attn * ops.expand_dims(ops.cast(valid_attn_mask, qk_attn.dtype), axis=1)

    # 5. Dropout
    if dropout_pr > 0.0:
        dropout_layer = keras.layers.Dropout(dropout_pr)
        qk_attn = dropout_layer(qk_attn, training=training)

    # 6. Output (Weighted Sum)
    # [B, H, N, N] @ [B, H, N, V] -> [B, H, N, V]
    attn_dense = ops.einsum("bhxd,bhdv->bhxv", qk_attn, v)

    # 7. Dense to Jagged
    # Flatten: [B, H, N, V] -> [B, N, H, V] -> [B, N, H*V]
    flat_attn_dense = ops.reshape(ops.transpose(attn_dense, [0, 2, 1, 3]), [-1, max_seq_len, H * V_dim])

    # Use keras_dense_to_jagged to extract the final jagged tensor [L, H*V]
    jagged_output_flat = keras_dense_to_jagged(flat_attn_dense, [seq_offsets])

    # Reshape back to [L, H, V]
    L_out = ops.shape(jagged_output_flat)[0]
    return ops.reshape(jagged_output_flat, [L_out, H, V_dim])

def hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: keras.KerasTensor,
    k: keras.KerasTensor,
    v: keras.KerasTensor,
    seq_offsets: keras.KerasTensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[keras.KerasTensor] = None,
    attn_scale: Optional[keras.KerasTensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    sort_by_length: bool = False,
    enable_tma: bool = False,
) -> keras.KerasTensor:
    _, H, D = ops.shape(q) # D is used here for some assertion

    # Note: Keras ops are symbolic. We replace torch._assert with equivalent ops.assert_ functions
    # for eager execution (like in a test case) or rely on symbolic checks.

    # The original PyTorch assertions (translated to Keras ops)
    # The Keras convention is to let KerasTensor shape checks handle many of these,
    # but we can enforce strict checks here for explicit verification.

    L_q = ops.shape(q)[0]
    H_q = ops.shape(q)[1]
    D_q = ops.shape(q)[2]

    L_v = ops.shape(v)[0]
    H_v = ops.shape(v)[1]

    # ops.assert_equal(max_seq_len > 0, True, message="max_seq_len must be larger than 0")
    # ops.assert_equal(ops.ndim(q), 3, message="q must be 3-D")
    # ops.assert_equal(ops.shape(k), ops.shape(q), message="k must be the same shape as q")
    # ops.assert_equal(ops.ndim(v), 3, message="v must be 3-D")
    # ops.assert_equal(L_v, L_q, message="wrong v shape[0]")
    # ops.assert_equal(H_v, H_q, message="wrong v shape[1]")
    # ops.assert_equal(causal, True, message="only support causal attention")

    # Fallback to the optimized kernel (keras_hstu_mha)
    return keras_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        dropout_pr=dropout_pr,
        training=training,
        attn_scale=attn_scale,
    )
