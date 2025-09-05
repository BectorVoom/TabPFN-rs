//  Copyright (c) Prior Labs GmbH 2025.

// TODO: Seems like there's a lot in this file that is over-parametrized for regular
// usage. Could likely just remove it.

use burn::{
    nn::LayerNorm as BurnLayerNorm,
    prelude::{Backend, Module, Tensor},
};
use std::marker::PhantomData;

use super::{
    attention::{Attention, full_attention::MultiHeadAttention},
    config::ModelConfig,
    mlp::MLP,
    transformer::DeterministicRngContext,
};

/// Create a causal mask that prevents attention to future positions
/// Returns tensor of shape [batch, seq_q, seq_kv] with 1.0 for allowed, 0.0 for masked
/// 
/// This mask ensures that position i can only attend to positions j where j <= i,
/// preventing information leakage from future time steps.
fn create_causal_mask<B: Backend>(
    batch_size: usize,
    seq_q: usize,
    seq_kv: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut mask_data = Vec::with_capacity(batch_size * seq_q * seq_kv);
    
    for _batch in 0..batch_size {
        for i in 0..seq_q {
            for j in 0..seq_kv {
                // Allow attention to current and previous positions only
                if j <= i {
                    mask_data.push(1.0f32);
                } else {
                    mask_data.push(0.0f32);
                }
            }
        }
    }
    
    Tensor::<B, 3>::from_floats(mask_data.as_slice(), device)
        .reshape([batch_size, seq_q, seq_kv])
}

/// Create a train/test separation mask that prevents test positions from attending to train labels
/// Returns tensor where train positions can attend everywhere, test positions only to previous positions
/// 
/// This mask is critical for preventing label leakage in in-context learning scenarios where
/// the model should not be able to see target values during evaluation.
fn create_train_test_separation_mask<B: Backend>(
    batch_size: usize,
    seq_q: usize,
    seq_kv: usize,
    single_eval_pos: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut mask_data = vec![1.0f32; batch_size * seq_q * seq_kv];
    
    // For each batch
    for batch_idx in 0..batch_size {
        for i in 0..seq_q {
            for j in 0..seq_kv {
                let idx = batch_idx * seq_q * seq_kv + i * seq_kv + j;
                
                // If current position i is in the test set (>= single_eval_pos)
                // and we're trying to attend to a position j that has target information
                if i >= single_eval_pos && j >= single_eval_pos && j > i {
                    mask_data[idx] = 0.0; // Block test->future_test attention
                }
            }
        }
    }
    
    Tensor::<B, 3>::from_floats(mask_data.as_slice(), device)
        .reshape([batch_size, seq_q, seq_kv])
}

// Constants removed - no longer needed after LayerNorm and MLP optimizations

/// Custom LayerNorm module that supports saving peak memory factor.
///
/// This module extends the Burn LayerNorm implementation to handle FP16 inputs
/// efficiently and support saving peak memory factor.
///
/// # Device Consistency
/// 
/// The LayerNorm module is created with a specific device and all operations
/// are performed on that device. Burn handles device placement automatically,
/// but the module stores device information for consistency validation when needed.
///
/// # Mathematical Properties
///
/// LayerNorm normalizes the input tensor over the last dimension(s) specified
/// by `normalized_shape`, producing output with zero mean and unit variance.
/// The normalization is applied as: `(x - mean) / sqrt(var + eps) * weight + bias`
/// where weight is initialized to 1 and bias to 0 by default.
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    /// The underlying Burn LayerNorm layer
    layer_norm: BurnLayerNorm<B>,
    /// Shape for normalization
    #[module(skip)]
    normalized_shape: Vec<usize>,
}

impl<B: Backend> LayerNorm<B> {
    /// Create a new LayerNorm with the given normalized shape and epsilon.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - The shape over which to normalize (typically the last dimension)
    /// * `eps` - Small value added to denominator for numerical stability
    /// * `_elementwise_affine` - Ignored (Burn's LayerNorm always has elementwise affine)
    /// * `device` - The device on which to create the LayerNorm module
    ///
    /// # Returns
    ///
    /// A new LayerNorm instance configured for the specified normalization shape
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: f64,
        _elementwise_affine: bool, // Burn's LayerNorm always has elementwise affine
        rng_ctx: &DeterministicRngContext<B>,
    ) -> Self {
        // Burn's LayerNorm expects a single dimension
        let d_model = normalized_shape[0];
        
        // Use deterministic LayerNorm creation for consistency
        // (Note: LayerNorm weights are deterministically initialized to 1 and bias to 0)
        let layer_norm = rng_ctx.create_deterministic_layer_norm(d_model, eps);

        Self {
            layer_norm,
            normalized_shape: normalized_shape.clone(),
        }
    }

    /// Compute the layer normalization.
    ///
    /// Applies the underlying Burn LayerNorm to the input tensor.
    fn compute(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // In Burn, the backend handles precision and optimizations automatically
        self.layer_norm.forward(x)
    }

    /// Perform layer normalization on the input tensor.
    ///
    /// Args:
    ///     input: The input tensor (3D).
    ///     allow_inplace: Whether to allow in-place operations (not used in Burn).
    ///     save_peak_mem_factor: The factor to save peak memory (not used directly).
    ///
    /// Returns:
    ///     The layer normalized tensor.
    ///
    /// # Panics
    /// Panics if the total number of elements is not divisible by the normalized shape elements.
    pub fn layernorm_forward_3d(
        &self,
        input: Tensor<B, 3>,
        _allow_inplace: bool,
        _save_peak_mem_factor: Option<i64>,
    ) -> Tensor<B, 3> {
        let input_shape = input.shape();
        let total_elements: usize = input_shape.dims.iter().product();
        let normalized_elements: usize = self.normalized_shape.iter().product();

        // Validate divisibility before reshape
        assert!(
            total_elements % normalized_elements == 0,
            "LayerNorm shape mismatch: total elements {} not divisible by normalized shape elements {}. Input shape: {:?}, normalized_shape: {:?}",
            total_elements,
            normalized_elements,
            input_shape.dims,
            self.normalized_shape
        );

        // Reshape to match LayerNorm requirements: [batch_size, normalized_shape...]
        let batch_size = total_elements / normalized_elements;
        let x = input.reshape([batch_size, normalized_elements]);

        let x = self.compute(x);

        // Reshape back to original 3D shape
        x.reshape([input_shape.dims[0], input_shape.dims[1], input_shape.dims[2]])
    }

    /// Perform layer normalization on the input tensor.
    ///
    /// Args:
    ///     input: The input tensor (4D).
    ///     allow_inplace: Whether to allow in-place operations (not used in Burn).
    ///     save_peak_mem_factor: The factor to save peak memory (not used directly).
    ///
    /// Returns:
    ///     The layer normalized tensor.
    ///
    /// # Panics
    /// Panics if the total number of elements is not divisible by the normalized shape elements.
    pub fn layernorm_forward_4d(
        &self,
        input: Tensor<B, 4>,
        _allow_inplace: bool,
        _save_peak_mem_factor: Option<i64>,
    ) -> Tensor<B, 4> {
        let input_shape = input.shape();
        let total_elements: usize = input_shape.dims.iter().product();
        let normalized_elements: usize = self.normalized_shape.iter().product();

        // Validate divisibility before reshape
        assert!(
            total_elements % normalized_elements == 0,
            "LayerNorm shape mismatch: total elements {} not divisible by normalized shape elements {}. Input shape: {:?}, normalized_shape: {:?}",
            total_elements,
            normalized_elements,
            input_shape.dims,
            self.normalized_shape
        );

        // Reshape to match LayerNorm requirements: [batch_size, normalized_shape...]
        let batch_size = total_elements / normalized_elements;
        let x = input.reshape([batch_size, normalized_elements]);

        let x = self.compute(x);

        // Reshape back to original 4D shape
        x.reshape([input_shape.dims[0], input_shape.dims[1], input_shape.dims[2], input_shape.dims[3]])
    }

    /// Validate that an input tensor is compatible with this LayerNorm
    /// In Burn, device consistency is typically handled automatically by the framework
    pub fn validate_tensor_compatibility<const D: usize>(&self, _input: &Tensor<B, D>) -> Result<(), String> {
        // Burn handles device consistency automatically in most cases
        // This method exists for future extensibility if needed
        Ok(())
    }
}

/// Transformer encoder layer that processes each feature block separately.
///
/// This layer consists of multi-head attention between features, multi-head
/// attention between items, and feedforward neural networks (MLPs).
///
/// It supports various configurations and optimization options.
#[derive(Module, Debug)]
pub struct PerFeatureEncoderLayer<B: Backend> {
    /// Attention between features (optional)
    self_attn_between_features: Option<MultiHeadAttention<B>>,
    /// Attention between items
    self_attn_between_items: MultiHeadAttention<B>,
    /// Primary MLP
    mlp: MLP<B>,
    /// Optional second MLP
    second_mlp: Option<MLP<B>>,
    /// Layer normalization modules (3 or 4 depending on second_mlp)
    layer_norms: Vec<LayerNorm<B>>,

    // Configuration fields (not trainable parameters)
    #[module(skip)]
    pre_norm: bool,
    #[module(skip)]
    save_peak_mem_factor: Option<i64>,
    #[module(skip)]
    multiquery_item_attention_for_test_set: bool,
    #[module(skip)]
    activation: String,
    #[module(skip)]
    emsize: usize,
    #[module(skip)]
    dim_feedforward: usize,
    #[module(skip)]
    zero_init: bool,
    #[module(skip)]
    recompute_attn: bool,
    #[module(skip)]
    _phantom: PhantomData<B>,
}

impl<B: Backend> PerFeatureEncoderLayer<B> {
    /// Create a new PerFeatureEncoderLayer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &ModelConfig,
        dim_feedforward: usize,
        activation: String,
        layer_norm_eps: f64,
        pre_norm: bool,
        device: &B::Device,
        second_mlp: bool,
        layer_norm_with_elementwise_affine: bool,
        zero_init: bool,
        save_peak_mem_factor: Option<i64>,
        attention_between_features: bool,
        d_k: Option<usize>,
        d_v: Option<usize>,
        precomputed_kv: Option<(Option<Tensor<B, 4>>, Option<Tensor<B, 4>>)>,
        rng_ctx: &DeterministicRngContext<B>,
        seed_offset: u64,
    ) -> Result<Self, String> {
        // Validate configuration
        if config.emsize as usize % config.nhead as usize != 0 && (d_k.is_none() || d_v.is_none()) {
            return Err("config.emsize must be divisible by config.nhead if d_k and d_v are not provided".to_string());
        }

        if config.multiquery_item_attention_for_test_set && config.multiquery_item_attention {
            return Err(
                "Cannot use both multiquery_item_attention_for_test_set and multiquery_item_attention".to_string()
            );
        }

        let d_k = d_k.unwrap_or(config.emsize as usize / config.nhead as usize);
        let d_v = d_v.unwrap_or(config.emsize as usize / config.nhead as usize);

        // Create attention between features if requested
        let self_attn_between_features = if attention_between_features {
            let _attn_type = get_feature_attn_constructor::<B>(config)?;
            Some(MultiHeadAttention::new(
                d_k,
                d_v,
                config,
                1, // share_kv_across_n_heads
                None, // dropout_p
                None, // softmax_scale
                zero_init,
                None, // precomputed_k
                None, // precomputed_v
                None, // precomputed_kv
                rng_ctx,
                seed_offset + 500, // init_seed_offset for features attention
                false, // inference_mode
            ))
        } else {
            None
        };

        // Handle precomputed KV
        let (precomputed_k, precomputed_v, precomputed_kv_tensor) = match precomputed_kv {
            Some((k_opt, v_opt)) => (k_opt, v_opt, None),
            None => (None, None, None),
        };

        // Create attention between items
        let _attn_type = get_item_attn_constructor::<B>(config)?;
        let self_attn_between_items = MultiHeadAttention::new(
            d_k,
            d_v,
            config,
            if config.multiquery_item_attention {
                config.nhead as usize
            } else {
                1
            },
            None, // dropout_p
            None, // softmax_scale
            zero_init,
            precomputed_k,
            precomputed_v,
            precomputed_kv_tensor,
            rng_ctx,
            seed_offset + 600, // init_seed_offset for items attention
            false, // inference_mode
        );

        // Create MLP using the provided device (consistent device usage)
        let (mlp, _mlp_config) = MLP::new_with_str_activation(
            config.emsize as usize,
            dim_feedforward,
            &activation,
            rng_ctx,
            seed_offset + 300, // MLP seed offset
            zero_init,
            config.recompute_attn,
        )?;

        // Create second MLP if requested using the same device
        let (second_mlp, _second_mlp_config) = if second_mlp {
            let (mlp, _config) = MLP::new_with_str_activation(
                config.emsize as usize,
                dim_feedforward,
                &activation,
                rng_ctx,
                seed_offset + 400, // Second MLP seed offset
                zero_init,
                config.recompute_attn,
            )?;
            (Some(mlp), Some(_config))
        } else {
            (None, None)
        };

        // Create layer norms (3 or 4 depending on second_mlp)
        let num_layer_norms = if second_mlp.is_some() { 4 } else { 3 };
        let mut layer_norms = Vec::with_capacity(num_layer_norms);

        for i in 0..num_layer_norms {
            layer_norms.push(LayerNorm::new(
                vec![config.emsize as usize],
                layer_norm_eps,
                layer_norm_with_elementwise_affine,
                rng_ctx,
            ));
        }

        // Device consistency note: Burn handles device placement automatically
        // All components (LayerNorms, MLPs, Attention) are created with the same device parameter
        // and Burn ensures consistent device usage across operations

        Ok(Self {
            self_attn_between_features,
            self_attn_between_items,
            mlp,
            second_mlp,
            layer_norms,
            pre_norm,
            save_peak_mem_factor,
            multiquery_item_attention_for_test_set: config.multiquery_item_attention_for_test_set,
            activation: activation.clone(),
            emsize: config.emsize as usize,
            dim_feedforward,
            zero_init,
            recompute_attn: config.recompute_attn,
            _phantom: PhantomData,
        })
    }

    /// Pass the input through the encoder layer.
    ///
    /// Args:
    ///     state: The transformer state of shape (batch_size, num_items, num_feature_blocks, d_model).
    ///     single_eval_pos: Position from which everything is treated as test set.
    ///     cache_trainset_representation: Whether to cache the trainset representation.
    ///     att_src: Optional tensor to attend to from the final encoder layer.
    ///     train: Whether to use training mode (enables dropout and other stochastic operations).
    ///
    /// Returns:
    ///     The transformer state passed through the encoder layer.
    pub fn encoder_forward(
        &mut self,
        state: Tensor<B, 4>,
        single_eval_pos: usize,
        cache_trainset_representation: bool,
        att_src: Option<Tensor<B, 4>>,
        train: bool,
    ) -> Tensor<B, 4> {
        // Validate input shape
        let state_shape = state.shape().dims;
        if state_shape.len() != 4 {
            panic!(
                "src must be of shape (batch_size, num_items, num_feature_blocks, d_model), got {:?}",
                state_shape
            );
        }

        let mut save_peak_mem_factor = self.save_peak_mem_factor;
        if cache_trainset_representation && single_eval_pos == 0 {
            assert!(self.self_attn_between_items.has_cached_kv());
            save_peak_mem_factor = None;
        }

        // Validate constraints
        if let Some(ref _att_src) = att_src {
            assert!(
                !self.multiquery_item_attention_for_test_set,
                "Not implemented yet."
            );
            assert!(!cache_trainset_representation, "Not implemented yet.");
            assert!(
                single_eval_pos == 0,
                "single_eval_pos should not be set, as the train representation is in att_src"
            );
        }

        if self.self_attn_between_features.is_none() {
            assert!(!cache_trainset_representation, "Not implemented yet.");
            assert!(
                state_shape[2] == 1,
                "One group architecture expects one feature group, but got {} feature groups.",
                state_shape[2]
            );
        }

        // Calculate MLP save peak memory factor
        let mlp_save_peak_mem_factor = save_peak_mem_factor.map(|factor| factor * 8);

        // Build sublayers in order
        let mut sublayers: Vec<(&str, usize)> = Vec::new();
        let mut layer_norm_idx = 0;

        // Add attention between features if available
        if self.self_attn_between_features.is_some() {
            sublayers.push(("features_attention", layer_norm_idx));
            layer_norm_idx += 1;
        } else {
            assert!(
                state_shape[2] == 1,
                "If there is no attention between features, the number of feature blocks must be 1."
            );
        }

        // Add second MLP if available
        if self.second_mlp.is_some() {
            sublayers.push(("second_mlp", layer_norm_idx));
            layer_norm_idx += 1;
        }

        // Add remaining sublayers
        sublayers.push(("items_attention", layer_norm_idx));
        layer_norm_idx += 1;
        sublayers.push(("primary_mlp", layer_norm_idx));

        // Process through sublayers
        let mut current_state = state;

        for (_sublayer_idx, (sublayer_type, norm_idx)) in sublayers.iter().enumerate() {
            // Pre-norm (currently disabled with assertion)
            if self.pre_norm {
                panic!(
                    "Pre-norm implementation is wrong, as the residual should never be layer normed here."
                );
            }

            // Apply sublayer
            current_state = match *sublayer_type {
                "features_attention" => {
                    // Attention between features
                    self.apply_attention_between_features(
                        current_state,
                        save_peak_mem_factor,
                        train,
                    )
                }
                "second_mlp" => {
                    // Second MLP
                    self.apply_second_mlp(
                        current_state,
                        mlp_save_peak_mem_factor,
                    )
                }
                "items_attention" => {
                    // Attention between items
                    self.apply_attention_between_items(
                        current_state,
                        single_eval_pos,
                        cache_trainset_representation,
                        att_src.as_ref(),
                        save_peak_mem_factor,
                        train,
                    )
                }
                "primary_mlp" => {
                    // Primary MLP
                    self.apply_primary_mlp(
                        current_state,
                        mlp_save_peak_mem_factor,
                    )
                }
                _ => panic!("Invalid sublayer type: {}", sublayer_type),
            };

            // Post-norm
            if !self.pre_norm {
                current_state = self.layer_norms[*norm_idx].layernorm_forward_4d(
                    current_state,
                    true, // allow_inplace
                    save_peak_mem_factor,
                );
            }
        }

        current_state
    }

    fn apply_attention_between_features(
        &mut self,
        x: Tensor<B, 4>,
        save_peak_mem_factor: Option<i64>,
        train: bool,
    ) -> Tensor<B, 4> {
        if let Some(ref mut attn) = self.self_attn_between_features {
            // Input shape: [batch_size, num_items, num_feature_blocks, d_model]
            // We need to transform this to work with attention which expects 3D tensors

            let x_shape = x.shape().dims;
            let [batch_size, num_items, num_feature_blocks, d_model] =
                [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];

            // For attention between features, we need to process each item separately
            // since attention expects [batch, seq, d_model] where d_model matches emsize
            // We'll apply attention across feature blocks for each item position

            let mut item_results = Vec::new();

            for item_idx in 0..num_items {
                // Extract one item across all features: [batch, features, d_model]
                let x_item = x.clone().slice([
                    0..batch_size,
                    item_idx..(item_idx + 1),
                    0..num_feature_blocks,
                    0..d_model,
                ]).squeeze::<3>(1); // Remove the item dimension: [batch, features, d_model]

                // Create causal mask for features to prevent future feature leakage
                let causal_mask = create_causal_mask::<B>(
                    batch_size,
                    num_feature_blocks,
                    num_feature_blocks,
                    &x_item.device(),
                );

                // Apply attention across features for this item
                let attended_item = attn.forward(
                    x_item,
                    None, // x_kv (self-attention)
                    false, // cache_kv
                    false, // use_cached_kv
                    false, // reuse_first_head_kv
                    false, // only_cache_first_head_kv
                    save_peak_mem_factor,
                    true, // add_input (residual connection)
                    true, // allow_inplace
                    train, // train mode for dropout control
                    Some(causal_mask), // attention_mask - prevent future feature leakage
                );

                // Add back the item dimension: [batch, 1, features, d_model]
                let attended_4d = attended_item.unsqueeze_dim(1);
                item_results.push(attended_4d);
            }

            // Concatenate all items back together: [batch, items, features, d_model]
            if item_results.len() == 1 {
                item_results.into_iter().next().unwrap()
            } else {
                Tensor::cat(item_results, 1)
            }
        } else {
            panic!("Attention between features is None but was called");
        }
    }

    fn apply_attention_between_items(
        &mut self,
        x: Tensor<B, 4>,
        single_eval_pos: usize,
        cache_trainset_representation: bool,
        att_src: Option<&Tensor<B, 4>>,
        save_peak_mem_factor: Option<i64>,
        train: bool,
    ) -> Tensor<B, 4> {
        // We need to transpose as self attention always treats dim -2 as the sequence dimension
        if self.multiquery_item_attention_for_test_set {
            self.apply_multiquery_attention_between_items(
                x,
                single_eval_pos,
                cache_trainset_representation,
                save_peak_mem_factor,
                train,
            )
        } else {
            self.apply_standard_attention_between_items(
                x,
                single_eval_pos,
                cache_trainset_representation,
                att_src,
                save_peak_mem_factor,
                train,
            )
        }
    }

    fn apply_multiquery_attention_between_items(
        &mut self,
        x: Tensor<B, 4>,
        single_eval_pos: usize,
        cache_trainset_representation: bool,
        save_peak_mem_factor: Option<i64>,
        train: bool,
    ) -> Tensor<B, 4> {
        let x_shape = x.shape().dims;
        let [batch_size, _num_items, num_feature_blocks, d_model] =
            [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];
        let mut result_parts = Vec::new();

        // Handle test set
        if single_eval_pos < x_shape[1] {
            let x_test = x.clone().slice([
                0..x_shape[0],
                single_eval_pos..x_shape[1],
                0..x_shape[2],
                0..x_shape[3],
            ]);

            let kv_src = if single_eval_pos > 0 {
                Some(x.clone().slice([
                    0..x_shape[0],
                    0..single_eval_pos,
                    0..x_shape[2],
                    0..x_shape[3],
                ]))
            } else {
                None
            };

            // Transpose for attention: (batch, items, features, d_model) -> (batch, features, items, d_model)
            let x_test_transposed = x_test.swap_dims(1, 2);
            let kv_src_transposed = kv_src.as_ref().map(|kv| kv.clone().swap_dims(1, 2));

            // Process each feature block separately
            let mut test_result_blocks = Vec::new();

            for feature_idx in 0..num_feature_blocks {
                // Extract test feature block: [batch, items, d_model]
                let x_test_feature = x_test_transposed.clone().slice([
                    0..batch_size,
                    feature_idx..(feature_idx + 1),
                    0..(x_shape[1] - single_eval_pos),
                    0..d_model,
                ]).squeeze::<3>(1);

                // Extract KV for this feature block if available
                let x_kv_feature = kv_src_transposed.as_ref().map(|src| {
                    src.clone().slice([
                        0..batch_size,
                        feature_idx..(feature_idx + 1),
                        0..single_eval_pos,
                        0..d_model,
                    ]).squeeze::<3>(1)
                });

                // Create train/test separation mask to prevent label leakage
                let seq_q = x_shape[1] - single_eval_pos;
                let seq_kv = if kv_src_transposed.is_some() { 
                    single_eval_pos + seq_q 
                } else { 
                    seq_q 
                };
                let test_separation_mask = create_train_test_separation_mask::<B>(
                    batch_size,
                    seq_q,
                    seq_kv,
                    0, // All test positions in this context
                    &x_test_feature.device(),
                );

                // Apply attention with multiquery settings
                let attended_test_feature = self.self_attn_between_items.forward(
                    x_test_feature,
                    x_kv_feature,
                    false, // cache_kv
                    single_eval_pos != 0, // use_cached_kv when single_eval_pos != 0
                    true, // reuse_first_head_kv
                    false, // only_cache_first_head_kv
                    save_peak_mem_factor,
                    true, // add_input (residual connection)
                    true, // allow_inplace
                    train, // train mode for dropout control
                    Some(test_separation_mask), // attention_mask - prevent test->future_test label leakage
                );

                // Add dimension back: [batch, 1, items, d_model]
                let attended_4d = attended_test_feature.unsqueeze_dim(1);
                test_result_blocks.push(attended_4d);
            }

            // Concatenate feature blocks and transpose back
            let test_result_transposed = if test_result_blocks.len() == 1 {
                test_result_blocks.into_iter().next().unwrap()
            } else {
                Tensor::cat(test_result_blocks, 1)
            };
            let new_x_test = test_result_transposed.swap_dims(1, 2);

            result_parts.push(new_x_test);
        }

        // Handle training set
        if single_eval_pos > 0 {
            let x_train = x.clone().slice([
                0..x_shape[0],
                0..single_eval_pos,
                0..x_shape[2],
                0..x_shape[3],
            ]);

            // Transpose for attention
            let x_train_transposed = x_train.swap_dims(1, 2);

            // Process each feature block separately
            let mut train_result_blocks = Vec::new();

            for feature_idx in 0..num_feature_blocks {
                // Extract train feature block: [batch, items, d_model]
                let x_train_feature = x_train_transposed.clone().slice([
                    0..batch_size,
                    feature_idx..(feature_idx + 1),
                    0..single_eval_pos,
                    0..d_model,
                ]).squeeze::<3>(1);

                // Create causal mask for training set to prevent future information leakage
                let train_causal_mask = create_causal_mask::<B>(
                    batch_size,
                    single_eval_pos,
                    single_eval_pos,
                    &x_train_feature.device(),
                );

                // Apply self-attention
                let attended_train_feature = self.self_attn_between_items.forward(
                    x_train_feature.clone(),
                    Some(x_train_feature), // self-attention (x_kv same as x)
                    cache_trainset_representation,
                    false, // use_cached_kv
                    false, // reuse_first_head_kv
                    true, // only_cache_first_head_kv
                    save_peak_mem_factor,
                    true, // add_input (residual connection)
                    true, // allow_inplace
                    train, // train mode for dropout control
                    Some(train_causal_mask), // attention_mask - prevent future information leakage in training
                );

                // Add dimension back: [batch, 1, items, d_model]
                let attended_4d = attended_train_feature.unsqueeze_dim(1);
                train_result_blocks.push(attended_4d);
            }

            // Concatenate feature blocks and transpose back
            let train_result_transposed = if train_result_blocks.len() == 1 {
                train_result_blocks.into_iter().next().unwrap()
            } else {
                Tensor::cat(train_result_blocks, 1)
            };
            let new_x_train = train_result_transposed.swap_dims(1, 2);

            result_parts.insert(0, new_x_train);
        }

        // Concatenate results
        if result_parts.len() == 1 {
            result_parts.into_iter().next().unwrap()
        } else {
            Tensor::cat(result_parts, 1)
        }
    }

    fn apply_standard_attention_between_items(
        &mut self,
        x: Tensor<B, 4>,
        single_eval_pos: usize,
        cache_trainset_representation: bool,
        att_src: Option<&Tensor<B, 4>>,
        save_peak_mem_factor: Option<i64>,
        train: bool,
    ) -> Tensor<B, 4> {
        let x_shape = x.shape().dims;
        let [batch_size, num_items, num_feature_blocks, d_model] =
            [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];

        // Transpose for attention: (batch, items, features, d_model) -> (batch, features, items, d_model)
        let x_transposed = x.clone().swap_dims(1, 2);

        // Prepare attention source (KV)
        let attention_src_x = if let Some(att_src_tensor) = att_src {
            Some(att_src_tensor.clone().swap_dims(1, 2))
        } else if single_eval_pos > 0 {
            // Use training data as KV source
            Some(x.clone().slice([
                0..x_shape[0],
                0..single_eval_pos,
                0..x_shape[2],
                0..x_shape[3],
            ]).swap_dims(1, 2))
        } else {
            None
        };

        // Process each feature block separately
        let mut result_blocks = Vec::new();

        for feature_idx in 0..num_feature_blocks {
            // Extract feature block: [batch, items, d_model]
            let x_feature = x_transposed.clone().slice([
                0..batch_size,
                feature_idx..(feature_idx + 1),
                0..num_items,
                0..d_model,
            ]).squeeze::<3>(1);

            // Extract KV for this feature block if available
            let x_kv_feature = attention_src_x.as_ref().map(|src| {
                src.clone().slice([
                    0..batch_size,
                    feature_idx..(feature_idx + 1),
                    0..src.shape().dims[2], // Use source sequence length
                    0..d_model,
                ]).squeeze::<3>(1)
            });

            // Create appropriate attention mask based on context
            let attention_mask = if single_eval_pos > 0 && single_eval_pos < num_items {
                // Mixed train/test scenario - create separation mask
                let seq_kv = if x_kv_feature.is_some() {
                    attention_src_x.as_ref().map(|src| src.shape().dims[2]).unwrap_or(num_items)
                } else {
                    num_items
                };
                Some(create_train_test_separation_mask::<B>(
                    batch_size,
                    num_items,
                    seq_kv,
                    single_eval_pos,
                    &x_feature.device(),
                ))
            } else if single_eval_pos == 0 {
                // Pure training scenario - use causal mask
                Some(create_causal_mask::<B>(
                    batch_size,
                    num_items,
                    num_items,
                    &x_feature.device(),
                ))
            } else {
                // Pure test scenario - use causal mask
                Some(create_causal_mask::<B>(
                    batch_size,
                    num_items,
                    num_items,
                    &x_feature.device(),
                ))
            };

            // Apply attention
            let attended_feature = self.self_attn_between_items.forward(
                x_feature,
                x_kv_feature,
                cache_trainset_representation && single_eval_pos > 0,
                cache_trainset_representation && single_eval_pos == 0,
                false, // reuse_first_head_kv
                false, // only_cache_first_head_kv
                save_peak_mem_factor,
                true, // add_input (residual connection)
                true, // allow_inplace
                train, // train mode for dropout control
                attention_mask, // attention_mask - prevent label leakage
            );

            // Add dimension back: [batch, 1, items, d_model]
            let attended_4d = attended_feature.unsqueeze_dim(1);
            result_blocks.push(attended_4d);
        }

        // Concatenate feature blocks back together
        let result_transposed = if result_blocks.len() == 1 {
            result_blocks.into_iter().next().unwrap()
        } else {
            Tensor::cat(result_blocks, 1)
        };

        // Transpose back: (batch, features, items, d_model) -> (batch, items, features, d_model)
        result_transposed.swap_dims(1, 2)
    }

    fn apply_primary_mlp(
        &mut self,
        x: Tensor<B, 4>,
        mlp_save_peak_mem_factor: Option<i64>,
    ) -> Tensor<B, 4> {
        let should_use_mem_factor = if let Some(factor) = mlp_save_peak_mem_factor {
            let total_elements = x.shape().dims.iter().product::<usize>();
            let last_dim = x.shape().dims[x.shape().dims.len() - 1];
            (total_elements / last_dim / factor as usize) > 32
        } else {
            false
        };

        let mem_factor = if should_use_mem_factor {
            mlp_save_peak_mem_factor.map(|f| f as usize)
        } else {
            None
        };

        // Create MLP config directly (avoiding device inconsistency from new_with_str_activation)
        let activation = match self.activation.as_str() {
            "GELU" => super::mlp::Activation::GELU,
            "RELU" => super::mlp::Activation::RELU,
            _ => super::mlp::Activation::GELU, // Default fallback
        };
        
        let config = super::mlp::MLPConfig {
            activation,
            recompute: self.recompute_attn,
        };

        // Apply MLP with memory optimization
        self.mlp.mlp_forward(x, &config, true, true, mem_factor)
    }

    fn apply_second_mlp(
        &mut self,
        x: Tensor<B, 4>,
        mlp_save_peak_mem_factor: Option<i64>,
    ) -> Tensor<B, 4> {
        if let Some(ref mut second_mlp) = self.second_mlp {
            let mem_factor = mlp_save_peak_mem_factor.map(|f| f as usize);

            // Create MLP config directly (avoiding device inconsistency from new_with_str_activation)
            let activation = match self.activation.as_str() {
                "GELU" => super::mlp::Activation::GELU,
                "RELU" => super::mlp::Activation::RELU,
                _ => super::mlp::Activation::GELU, // Default fallback
            };
            
            let config = super::mlp::MLPConfig {
                activation,
                recompute: self.recompute_attn,
            };

            second_mlp.mlp_forward(x, &config, true, true, mem_factor)
        } else {
            panic!("Second MLP is None but was called");
        }
    }

    /// Empty the trainset representation cache.
    pub fn empty_trainset_representation_cache(&mut self) {
        self.self_attn_between_items.empty_kv_cache();

        if let Some(ref mut attn) = self.self_attn_between_features {
            attn.empty_kv_cache(); // not necessary, just in case
        }
    }
}

/// Get the feature attention constructor based on configuration
pub fn get_feature_attn_constructor<B: Backend>(
    config: &ModelConfig,
) -> Result<String, String> {
    match config.feature_attention_type.as_str() {
        "full" => Ok("full".to_string()),
        _ => Err(format!("Unknown attention type: {}", config.feature_attention_type)),
    }
}

/// Get the item attention constructor based on configuration  
pub fn get_item_attn_constructor<B: Backend>(
    config: &ModelConfig,
) -> Result<String, String> {
    match config.item_attention_type.as_str() {
        "full" => Ok("full".to_string()),
        _ => Err(format!("Unknown attention type: {}", config.item_attention_type)),
    }
}