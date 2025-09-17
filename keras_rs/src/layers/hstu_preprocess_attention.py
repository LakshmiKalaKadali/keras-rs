Python 3.12.5 (v3.12.5:ff3bc82f7c9, Aug  7 2024, 05:32:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
... 
... def keras_hstu_preprocess_and_attention(
...     x, norm_weight, norm_bias, norm_eps, num_heads, attn_dim, hidden_dim,
...     uvqk_weight, uvqk_bias, max_seq_len, seq_offsets, attn_alpha, causal,
...     num_targets, max_attn_len, contextual_seq_len, recompute_uvqk_in_backward,
...     recompute_normed_x_in_backward, sort_by_length, prefill = False,
...     kernel = HammerKernel.KERAS, **kwargs
... ) -> Tuple:
... 
...     L_x, D_in = ops.shape(x)
...     UVQK_OUT_CHECK = 2 * num_heads * (hidden_dim + attn_dim)
... 
...     # --- Assertions (Converted from torch._assert to ops.assert_...) ---
... 
...     # Assertions are functional, usually only checked when the model is traced/compiled.
...     # We use standard Python asserts for the most basic checks on constant inputs.
...     assert max_seq_len > 0, "max_seq_len must be larger than 0"
...     assert ops.ndim(x) == 2, "x must be 2-D"
...     assert causal is True, "only causal attention is supported."
... 
...     # Check shapes using ops.assert_equal for dynamic tracing compliance
...     #
... 
...     # --- Skip Triton Path and use Keras Path ---
...     if kernel in [HammerKernel.TRITON, HammerKernel.TRITON_CC] and prefill is False:
...         # The Triton path is skipped for this Keras test environment
...         raise NotImplementedError("Triton path not mocked/supported for this test.")
...     else:
...         # 1. Compute U, Q, K, V
...         u, q, k, v = keras_hstu_compute_uqvk(
...             x=x, norm_weight=norm_weight, norm_bias=norm_bias, norm_eps=norm_eps,
            num_heads=num_heads, attn_dim=attn_dim, hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight, uvqk_bias=uvqk_bias, kernel=kernel,
        )

        # 2. Compute Attention
        attn_output = keras_hstu_mha(
            max_seq_len=max_seq_len, alpha=attn_alpha, q=q, k=k, v=v,
            seq_offsets=seq_offsets, causal=causal, dropout_pr=0.0,
            training=False, num_targets=num_targets, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len, sort_by_length=sort_by_length,
            kernel=kernel, **kwargs
        )

        # Reshape: [L, H, D] -> [L, H * D]
        attn_output = ops.reshape(attn_output, [-1, hidden_dim * num_heads])

        # Returns u, attention output, k, v
        return u, attn_output, k, v

