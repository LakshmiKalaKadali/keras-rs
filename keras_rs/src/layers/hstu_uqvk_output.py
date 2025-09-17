Python 3.12.5 (v3.12.5:ff3bc82f7c9, Aug  7 2024, 05:32:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> def keras_layer_norm(
...     x: keras.KerasTensor,
...     weight: keras.KerasTensor,
...     bias: keras.KerasTensor,
...     eps: float,
...     kernel: HammerKernel,
... ) -> keras.KerasTensor:
...     """
...     Keras 3 implementation of Layer Normalization.
...     Note: Keras LayerNormalization doesn't take 'weight'/'bias' tensors directly.
...     We simulate the scale/bias application externally as done in the PyTorch F.layer_norm.
...     """
...     # 1. Normalize x
...     mean = ops.mean(x, axis=-1, keepdims=True)
...     variance = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
...     x_norm = (x - mean) / ops.sqrt(variance + eps)
... 
...     # 2. Apply weight and bias (Gamma * x_norm + Beta)
...     return x_norm * weight + bias
... 
... def keras_addmm(
...     bias: keras.KerasTensor,
...     input: keras.KerasTensor,
...     mat2: keras.KerasTensor,
...     kernel: HammerKernel,
... ) -> keras.KerasTensor:
...     """Keras 3 equivalent of torch.addmm (bias + input @ mat2)."""
...     return ops.add(bias, ops.matmul(input, mat2))
... 
... def hstu_compute_uqvk(
...     x: keras.KerasTensor,
...     norm_weight: keras.KerasTensor,
...     norm_bias: keras.KerasTensor,
...     norm_eps: float,
...     num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: keras.KerasTensor,
    uvqk_bias: keras.KerasTensor,
    kernel: HammerKernel = HammerKernel.KERAS,
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:

    normed_x = keras_layer_norm(
        x,
        weight=norm_weight,
        bias=norm_bias,
        eps=norm_eps,
        kernel=kernel,
    )

    # uvqk = bias + normed_x @ uvqk_weight
    uvqk = keras_addmm(uvqk_bias, normed_x, uvqk_weight, kernel)

    # Calculate split sizes
    u_size = hidden_dim * num_heads
    v_size = hidden_dim * num_heads
    q_size = attn_dim * num_heads
    k_size = attn_dim * num_heads

    # Use ops.slice to split the tensor along dim=1 (feature dimension)
    start_u = 0
    start_v = start_u + u_size
    start_q = start_v + v_size
    start_k = start_q + k_size

    # FIX: Changed 'size' to 'shape' for keras.ops.slice
    # The -1 in shape means "all elements" along the batch dimension
    u = ops.slice(uvqk, start_indices=[0, start_u], shape=[-1, u_size])
    v = ops.slice(uvqk, start_indices=[0, start_v], shape=[-1, v_size])
    q = ops.slice(uvqk, start_indices=[0, start_q], shape=[-1, q_size])
    k = ops.slice(uvqk, start_indices=[0, start_k], shape=[-1, k_size])

    # Apply SiLU (Swish) activation to U
    u = ops.silu(u)

    # Reshape Q, K, V from (B*L, Heads*Dim) to (B*L, Heads, Dim)
    q = ops.reshape(q, (-1, num_heads, attn_dim))
    k = ops.reshape(k, (-1, num_heads, attn_dim))
    v = ops.reshape(v, (-1, num_heads, hidden_dim))

    return u, q, k, v
