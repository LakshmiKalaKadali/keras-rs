Python 3.12.5 (v3.12.5:ff3bc82f7c9, Aug  7 2024, 05:32:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
def keras_norm_mul_dropout(
    x: keras.KerasTensor,
    u: keras.KerasTensor,
    weight: keras.KerasTensor,
    bias: keras.KerasTensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> keras.KerasTensor:
    """
    Keras 3 equivalent of pytorch_norm_mul_dropout.
    Applies normalization, element-wise multiplication with u, and dropout.
    """
    x = ops.convert_to_tensor(x, dtype='float32')
    u = ops.convert_to_tensor(u, dtype='float32')

    if silu_u:
        u = ops.silu(u)

    if group_norm:
        # Group norm is complex to implement generically with Keras ops;
        # For simplicity and correctness in this test, we mock the result
        # based on the LayerNorm path, as LayerNorm is the typical default.
        raise NotImplementedError("Group Norm path requires custom implementation not suitable for simple Keras ops conversion.")
    else:
        # Default path: Layer Normalization (using Keras Layer)
        norm_layer = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=eps,
            scale=False, # We apply scale/bias externally via multiplication
            center=False,
        )
        # Note: Keras LayerNormalization doesn't take 'weight'/'bias' tensors directly in the call.
        # It's typically part of the layer's variables. Since we are using Keras ops style,
        # we will simulate the behavior of F.layer_norm where scale/bias is applied externally

        # 1. Normalize x
        mean = ops.mean(x, axis=-1, keepdims=True)
        variance = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
        x_norm = (x - mean) / ops.sqrt(variance + eps)

        # 2. Apply weight and bias (Gamma * x_norm + Beta)
        y_norm = x_norm * weight + bias

        # 3. Apply u multiplication
        y = u * y_norm

    if concat_ux:
        y = ops.concatenate([u, x, y], axis=1)

    # Dropout (uses Keras layer for correct behavior, especially training=True/False)
    y = keras.layers.Dropout(dropout_ratio)(y, training=training)

    # Keras ops typically maintain high-precision, so we skip explicit dtype conversion
    return ops.cast(y, dtype=x.dtype)

def keras_hstu_compute_output(
    attn: keras.KerasTensor,
    u: keras.KerasTensor,
    x: keras.KerasTensor,
    norm_weight: keras.KerasTensor,
    norm_bias: keras.KerasTensor,
    output_weight: keras.KerasTensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> keras.KerasTensor:
    y = keras_norm_mul_dropout(
        x=attn,
        u=u,
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
        dropout_ratio=dropout_ratio,
        training=training,
        silu_u=silu_u,
        concat_ux=concat_ux,
        group_norm=group_norm,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )

    # Equivalent of torch.add(x, torch.matmul(y, output_weight))
    # Note: Keras ops handle batched matrix multiplication correctly.
    output = ops.add(x, ops.matmul(y, output_weight))

    return output

def hstu_compute_output(
...     attn: keras.KerasTensor,
...     u: keras.KerasTensor,
...     x: keras.KerasTensor,
...     norm_weight: keras.KerasTensor,
...     norm_bias: keras.KerasTensor,
...     norm_eps: float,
...     output_weight: keras.KerasTensor,
...     num_heads: int,
...     linear_dim: int,
...     dropout_ratio: float,
...     training: bool,
...     concat_ux: bool,
...     group_norm: bool,
...     recompute_y_in_backward: bool,
...     kernel: HammerKernel = HammerKernel.KERAS,
... ) -> keras.KerasTensor:
...     """
...     Top-level function. Only uses the Keras kernel path.
...     """
...     # Using the Keras implementation regardless of the kernel value,
...     # as the other kernels (Triton, PyTorch) are not implemented here.
...     return keras_hstu_compute_output(
...         attn=attn,
...         u=u,
...         x=x,
...         norm_weight=norm_weight,
...         norm_bias=norm_bias,
...         output_weight=output_weight,
...         eps=norm_eps,
...         dropout_ratio=dropout_ratio,
...         training=training,
...         silu_u=False,
...         concat_ux=concat_ux,
...         group_norm=group_norm,
...         num_heads=num_heads,
...         linear_dim=linear_dim,
...     )
... 
... 
