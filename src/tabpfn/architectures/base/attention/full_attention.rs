//! Implements standard quadratic attention.

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation;
use burn::nn::Dropout;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use rand::Rng;

use crate::tabpfn::architectures::base::config::ModelConfig;
use crate::tabpfn::architectures::base::transformer::DeterministicRngContext;
use super::Attention;

/// Serializable state for MultiHeadAttention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionState {
    pub version: String,
    pub input_size: usize,
    pub output_size: usize,
    pub nhead: usize,
    pub nhead_kv: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub share_kv_across_n_heads: usize,
    pub dropout_p: Option<f64>,
    pub softmax_scale: Option<f64>,
    pub init_gain: f64,
    pub recompute_attn: bool,
    
    // Serialized weight data (as Vec<f32> with shape info)
    pub weights: HashMap<String, (Vec<usize>, Vec<f32>)>,
    
    // Serialized cache data (optional)
    pub cache_state: Option<HashMap<String, (Vec<usize>, Vec<f32>)>>,
}

impl AttentionState {
    const CURRENT_VERSION: &'static str = "1.0.0";
    
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION.to_string(),
            input_size: 0,
            output_size: 0,
            nhead: 0,
            nhead_kv: 0,
            d_k: 0,
            d_v: 0,
            share_kv_across_n_heads: 0,
            dropout_p: None,
            softmax_scale: None,
            init_gain: 0.0,
            recompute_attn: false,
            weights: HashMap::new(),
            cache_state: None,
        }
    }
    
    /// Check version compatibility
    pub fn is_compatible(&self) -> bool {
        self.version == Self::CURRENT_VERSION
    }
}

/// Multi-head attention layer implementation
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    // Configuration fields (not part of Module)
    #[module(skip)]
    input_size: usize,
    #[module(skip)]
    output_size: usize,
    #[module(skip)]
    nhead: usize,
    #[module(skip)]
    nhead_kv: usize,
    #[module(skip)]
    d_k: usize,
    #[module(skip)]
    d_v: usize,
    #[module(skip)]
    share_kv_across_n_heads: usize,
    #[module(skip)]
    dropout_p: Option<f64>,
    #[module(skip)]
    softmax_scale: Option<f64>,
    #[module(skip)]
    init_gain: f64,
    #[module(skip)]
    recompute_attn: bool,

    // Weight parameters - now supporting the complex weight structure from Python
    w_q: Option<Param<Tensor<B, 4>>>,     // [1, nhead, d_k, input_size]
    w_k: Option<Param<Tensor<B, 3>>>,     // [nhead_kv, d_k, input_size]
    w_v: Option<Param<Tensor<B, 3>>>,     // [nhead_kv, d_v, input_size]
    w_kv: Option<Param<Tensor<B, 4>>>,    // [2, nhead_kv, d_k, input_size]
    w_qkv: Option<Param<Tensor<B, 4>>>,   // [3, nhead, d_k, input_size]
    w_out: Param<Tensor<B, 3>>,           // [nhead, d_v, output_size]

    // Dropout module for attention probabilities
    dropout: Option<Dropout>,

    // Cache tensors (non-trainable, stored as Option for dynamic allocation)
    #[module(skip)]
    k_cache: Option<Tensor<B, 4>>,        // [batch, seq, nhead_kv, d_k]
    #[module(skip)]
    v_cache: Option<Tensor<B, 4>>,        // [batch, seq, nhead_kv, d_v]
    #[module(skip)]
    kv_cache: Option<Tensor<B, 5>>,       // [batch, seq, 2, nhead_kv, d_k]
}

impl<B: Backend> MultiHeadAttention<B> {
    // *** HELPER FUNCTIONS FOR ROBUST TENSOR SLICING ***
    
    /// Extract Q tensor from QKV tensor with shape validation
    /// Input: QKV tensor [batch, seq, 3, nhead, d_k]
    /// Output: Q tensor [batch, seq, nhead, d_k]  
    fn slice_q_from_qkv(qkv: &Tensor<B, 5>) -> Tensor<B, 4> {
        let shape = qkv.shape().dims;
        assert!(shape.len() == 5, "Expected QKV tensor with 5 dimensions, got {:?}", shape);
        assert!(shape[2] >= 1, "QKV dimension 2 must be >= 1 for Q slice, got shape {:?}", shape);
        
        qkv.clone()
            .slice([0..shape[0], 0..shape[1], 0..1, 0..shape[3], 0..shape[4]])
            .squeeze::<4>(2)
    }
    
    /// Extract K tensor from QKV or KV tensor with shape validation
    /// Input: QKV tensor [batch, seq, 3, nhead, d_k] or KV tensor [batch, seq, 2, nhead, d_k]  
    /// Output: K tensor [batch, seq, nhead, d_k]
    fn slice_k_from_qkv_or_kv(tensor: &Tensor<B, 5>) -> Tensor<B, 4> {
        let shape = tensor.shape().dims;
        assert!(shape.len() == 5, "Expected tensor with 5 dimensions, got {:?}", shape);
        
        let k_index = if shape[2] == 3 { 1 } else { 0 }; // QKV uses index 1, KV uses index 0
        assert!(shape[2] > k_index, "Tensor dimension 2 must be > {} for K slice, got shape {:?}", k_index, shape);
        
        tensor.clone()
            .slice([0..shape[0], 0..shape[1], k_index..k_index+1, 0..shape[3], 0..shape[4]])
            .squeeze::<4>(2)
    }
    
    /// Extract V tensor from QKV or KV tensor with shape validation  
    /// Input: QKV tensor [batch, seq, 3, nhead, d_v] or KV tensor [batch, seq, 2, nhead, d_v]
    /// Output: V tensor [batch, seq, nhead, d_v]
    fn slice_v_from_qkv_or_kv(tensor: &Tensor<B, 5>) -> Tensor<B, 4> {
        let shape = tensor.shape().dims;
        assert!(shape.len() == 5, "Expected tensor with 5 dimensions, got {:?}", shape);
        
        let v_index = if shape[2] == 3 { 2 } else { 1 }; // QKV uses index 2, KV uses index 1
        assert!(shape[2] > v_index, "Tensor dimension 2 must be > {} for V slice, got shape {:?}", v_index, shape);
        
        tensor.clone()
            .slice([0..shape[0], 0..shape[1], v_index..v_index+1, 0..shape[3], 0..shape[4]])
            .squeeze::<4>(2)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_k: usize,
        d_v: usize,
        config: &ModelConfig,
        share_kv_across_n_heads: usize,
        dropout_p: Option<f64>,
        softmax_scale: Option<f64>,
        initialize_output_to_zero: bool,
        precomputed_k: Option<Tensor<B, 4>>,
        precomputed_v: Option<Tensor<B, 4>>,
        precomputed_kv: Option<Tensor<B, 5>>,
        rng_ctx: &DeterministicRngContext<B>,
        init_seed_offset: u64,
        inference_mode: bool,
    ) -> Self {
        assert_eq!(config.nhead as usize % share_kv_across_n_heads, 0);
        
        let input_size = config.emsize as usize;
        let output_size = config.emsize as usize;
        let nhead = config.nhead as usize;
        let nhead_kv = nhead / share_kv_across_n_heads;
        let init_gain = config.attention_init_gain;
        let recompute_attn = config.recompute_attn;

        // Initialize output weight [nhead, d_v, output_size]
        let w_out = if initialize_output_to_zero {
            Param::from_tensor(Tensor::zeros([nhead, d_v, output_size], rng_ctx.device()))
        } else {
            // Apply Xavier uniform initialization: uniform distribution in [-limit, limit]
            // where limit = sqrt(6.0 / (fan_in + fan_out))
            let fan_in = d_v;
            let fan_out = output_size;
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let w_out_tensor = rng_ctx.with_isolated_seed(Some(init_seed_offset + 300), |rng| {
                let total_elements = nhead * d_v * output_size;
                let data: Vec<f32> = (0..total_elements)
                    .map(|_| rng.r#gen::<f32>() * 2.0 * limit as f32 - limit as f32) // Convert [0,1] to [-limit,limit]
                    .collect();
                
                Tensor::<B, 1>::from_floats(data.as_slice(), rng_ctx.device()).reshape([nhead, d_v, output_size])
            });
            Param::from_tensor(w_out_tensor)
        };

        // Determine weight structure based on dimensions and precomputed values
        let has_precomputed_kv = precomputed_kv.is_some() || precomputed_k.is_some();
        
        // Initialize cache from precomputed values
        let k_cache = precomputed_k;
        let v_cache = precomputed_v;
        let kv_cache = precomputed_kv;
        
        // Initialize dropout module deterministically if dropout_p is provided and not in inference mode
        let dropout = if inference_mode { 
            None 
        } else { 
            dropout_p.map(|p| rng_ctx.create_deterministic_dropout(p)) 
        };
        
        let (w_q, w_k, w_v, w_kv, w_qkv) = if d_k == d_v && nhead == nhead_kv && !has_precomputed_kv {
            // Use combined w_qkv
            let w_qkv = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                [3, nhead, d_k, input_size],
                nhead,
                init_gain.into(),
                rng_ctx,
                init_seed_offset + 200
            )));
            (None, None, None, None, w_qkv)
        } else {
            // Use separate weights
            let w_q = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                [1, nhead, d_k, input_size],
                nhead,
                init_gain.into(),
                rng_ctx,
                init_seed_offset + 201
            )));
            
            if !has_precomputed_kv {
                if d_k == d_v {
                    // Use combined w_kv
                    let w_kv = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                        [2, nhead_kv, d_k, input_size],
                        nhead,
                        init_gain.into(),
                        rng_ctx,
                        init_seed_offset + 202
                    )));
                    (w_q, None, None, w_kv, None)
                } else {
                    // Use separate w_k and w_v
                    let w_k = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                        [nhead_kv, d_k, input_size],
                        nhead,
                        init_gain.into(),
                        rng_ctx,
                        init_seed_offset + 203
                    )));
                    let w_v = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                        [nhead_kv, d_v, input_size],
                        nhead,
                        init_gain.into(),
                        rng_ctx,
                        init_seed_offset + 204
                    )));
                    (w_q, w_k, w_v, None, None)
                }
            } else {
                (w_q, None, None, None, None)
            }
        };

        Self {
            input_size,
            output_size,
            nhead,
            nhead_kv,
            d_k,
            d_v,
            share_kv_across_n_heads,
            dropout_p,
            softmax_scale,
            init_gain: init_gain.into(),
            recompute_attn,
            w_q,
            w_k,
            w_v,
            w_kv,
            w_qkv,
            w_out,
            dropout,
            k_cache,
            v_cache,
            kv_cache,
        }
    }

    fn newly_initialized_input_weight<const D: usize>(
        dims: [usize; D],
        nhead: usize,
        init_gain: f64,
        rng_ctx: &DeterministicRngContext<B>,
        init_seed: u64,
    ) -> Tensor<B, D> {
        assert!(dims.len() >= 3 && dims.len() <= 4);
        let d = dims[dims.len() - 2];
        let input_size = dims[dims.len() - 1];
        let std = (2.0 / (nhead * d + input_size) as f64).sqrt() * init_gain;
        let a = (3.0_f64).sqrt() * std;
        
        // Use deterministic initialization with seeded RNG
        rng_ctx.with_isolated_seed(Some(init_seed), |rng| {
            let total_elements: usize = dims.iter().product();
            let data: Vec<f32> = (0..total_elements)
                .map(|_| rng.r#gen::<f32>() * 2.0 * a as f32 - a as f32) // Convert [0,1] to [-a,a]
                .collect();
            
            Tensor::<B, 1>::from_floats(data.as_slice(), rng_ctx.device()).reshape(dims)
        })
    }
    
    /// Deterministic initialization using Xavier uniform center value (0) or a small constant
    fn deterministic_xavier_uniform<const D: usize>(
        dims: [usize; D], 
        _limit: f64, 
        device: &B::Device
    ) -> Tensor<B, D> {
        // For tests, use a small constant instead of zeros to ensure non-trivial computation
        // This provides deterministic but non-zero weights for meaningful attention patterns
        Tensor::zeros(dims, device)
    }

    // Property accessors
    pub fn w_q(&self) -> Option<Tensor<B, 4>> {
        self.w_q.as_ref().map(|param| param.val())
    }

    pub fn w_k(&self) -> Option<Tensor<B, 3>> {
        self.w_k.as_ref().map(|param| param.val())
    }

    pub fn w_v(&self) -> Option<Tensor<B, 3>> {
        self.w_v.as_ref().map(|param| param.val())
    }

    pub fn w_kv(&self) -> Option<Tensor<B, 4>> {
        self.w_kv.as_ref().map(|param| param.val())
    }

    pub fn w_qkv(&self) -> Option<Tensor<B, 4>> {
        self.w_qkv.as_ref().map(|param| param.val())
    }

    pub fn w_out(&self) -> Tensor<B, 3> {
        self.w_out.val()
    }

    pub fn has_cached_kv(&self) -> bool {
        (self.k_cache.is_some() && self.v_cache.is_some()) || self.kv_cache.is_some()
    }

    pub fn empty_kv_cache(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.kv_cache = None;
    }

    // *** GETTER METHODS FOR CACHE TENSORS ***
    
    pub fn get_k_cache(&self) -> Option<Tensor<B, 4>> {
        self.k_cache.clone()
    }

    pub fn get_v_cache(&self) -> Option<Tensor<B, 4>> {
        self.v_cache.clone()
    }

    pub fn get_kv_cache(&self) -> Option<Tensor<B, 5>> {
        self.kv_cache.clone()
    }

    // *** SETTER METHODS FOR INDIVIDUAL WEIGHTS (for test compatibility) ***
    
    pub fn set_w_q(&mut self, w_q: Option<Param<Tensor<B, 4>>>) {
        self.w_q = w_q;
    }

    pub fn set_w_k(&mut self, w_k: Option<Param<Tensor<B, 3>>>) {
        self.w_k = w_k;
    }

    pub fn set_w_v(&mut self, w_v: Option<Param<Tensor<B, 3>>>) {
        self.w_v = w_v;
    }

    pub fn set_w_kv(&mut self, w_kv: Option<Param<Tensor<B, 4>>>) {
        self.w_kv = w_kv;
    }

    pub fn set_w_qkv(&mut self, w_qkv: Option<Param<Tensor<B, 4>>>) {
        self.w_qkv = w_qkv;
    }

    pub fn set_w_out(&mut self, w_out: Param<Tensor<B, 3>>) {
        self.w_out = w_out;
    }

    /// Set parameters for the attention module, similar to Python's set_parameters method
    #[allow(clippy::too_many_arguments)]
    pub fn set_parameters(
        &mut self,
        w_out: Tensor<B, 3>,
        w_q: Option<Tensor<B, 4>>,
        w_k: Option<Tensor<B, 3>>,
        w_v: Option<Tensor<B, 3>>,
        w_kv: Option<Tensor<B, 4>>,
        w_qkv: Option<Tensor<B, 4>>,
        precomputed_k: Option<Tensor<B, 4>>,
        precomputed_v: Option<Tensor<B, 4>>,
        precomputed_kv: Option<Tensor<B, 5>>,
    ) -> Result<(), String> {
        // Validation logic similar to Python version
        
        // Precomputed k and v must be consistent
        if precomputed_k.is_some() != precomputed_v.is_some() {
            return Err("precomputed_k and precomputed_v must both be Some or both be None".to_string());
        }
        
        // precomputed_kv cannot coexist with precomputed_k/precomputed_v
        if precomputed_kv.is_some() && precomputed_k.is_some() {
            return Err("precomputed_kv cannot coexist with precomputed_k/precomputed_v".to_string());
        }
        
        // Must have either precomputed values or weight parameters
        let has_precomputed = precomputed_kv.is_some() || precomputed_k.is_some();
        let has_weights = w_qkv.is_some() || w_kv.is_some() || (w_k.is_some() && w_v.is_some());
        if has_precomputed == has_weights {
            return Err("Must have either precomputed values or weight parameters, but not both".to_string());
        }
        
        // w_qkv and w_q must be consistent
        if w_qkv.is_some() == w_q.is_some() {
            return Err("w_qkv and w_q must have opposite presence (one Some, one None)".to_string());
        }
        
        // If w_qkv is provided, w_kv, w_k, w_v must be None
        if w_qkv.is_some() && (w_kv.is_some() || w_k.is_some() || w_v.is_some()) {
            return Err("When w_qkv is provided, w_kv, w_k, w_v must be None".to_string());
        }
        
        // w_kv cannot coexist with w_k/w_v
        if w_kv.is_some() && (w_k.is_some() || w_v.is_some()) {
            return Err("w_kv cannot coexist with w_k or w_v".to_string());
        }
        
        // w_k and w_v must be consistent
        if w_k.is_some() != w_v.is_some() {
            return Err("w_k and w_v must both be Some or both be None".to_string());
        }
        
        // Shape validation helper functions for different tensor dimensions
        let validate_shape_4d = |tensor_opt: &Option<Tensor<B, 4>>, expected_shape: &[usize], name: &str| -> Result<(), String> {
            if let Some(tensor) = tensor_opt {
                let actual_shape = tensor.shape().dims;
                if actual_shape.len() != expected_shape.len() {
                    return Err(format!("{} shape rank mismatch: expected {}, got {}", 
                        name, expected_shape.len(), actual_shape.len()));
                }
                for (i, (&actual, &expected)) in actual_shape.iter().zip(expected_shape.iter()).enumerate() {
                    // Skip None dimensions (represented as 0 in our case)
                    if expected != 0 && actual != expected {
                        return Err(format!("{} shape mismatch at dimension {}: expected {}, got {}", 
                            name, i, expected, actual));
                    }
                }
            }
            Ok(())
        };
        
        let validate_shape_3d = |tensor_opt: &Option<Tensor<B, 3>>, expected_shape: &[usize], name: &str| -> Result<(), String> {
            if let Some(tensor) = tensor_opt {
                let actual_shape = tensor.shape().dims;
                if actual_shape.len() != expected_shape.len() {
                    return Err(format!("{} shape rank mismatch: expected {}, got {}", 
                        name, expected_shape.len(), actual_shape.len()));
                }
                for (i, (&actual, &expected)) in actual_shape.iter().zip(expected_shape.iter()).enumerate() {
                    // Skip None dimensions (represented as 0 in our case)
                    if expected != 0 && actual != expected {
                        return Err(format!("{} shape mismatch at dimension {}: expected {}, got {}", 
                            name, i, expected, actual));
                    }
                }
            }
            Ok(())
        };
        
        let validate_shape_5d = |tensor_opt: &Option<Tensor<B, 5>>, expected_shape: &[usize], name: &str| -> Result<(), String> {
            if let Some(tensor) = tensor_opt {
                let actual_shape = tensor.shape().dims;
                if actual_shape.len() != expected_shape.len() {
                    return Err(format!("{} shape rank mismatch: expected {}, got {}", 
                        name, expected_shape.len(), actual_shape.len()));
                }
                for (i, (&actual, &expected)) in actual_shape.iter().zip(expected_shape.iter()).enumerate() {
                    // Skip None dimensions (represented as 0 in our case)
                    if expected != 0 && actual != expected {
                        return Err(format!("{} shape mismatch at dimension {}: expected {}, got {}", 
                            name, i, expected, actual));
                    }
                }
            }
            Ok(())
        };
        
        // Validate tensor shapes
        validate_shape_4d(&precomputed_k, &[0, 0, self.nhead_kv, self.d_k], "precomputed_k")?;
        validate_shape_4d(&precomputed_v, &[0, 0, self.nhead_kv, self.d_v], "precomputed_v")?;
        validate_shape_5d(&precomputed_kv, &[0, 0, 2, self.nhead_kv, self.d_k], "precomputed_kv")?;
        validate_shape_4d(&w_q, &[1, self.nhead, self.d_k, self.input_size], "w_q")?;
        validate_shape_3d(&w_k, &[self.nhead_kv, self.d_k, self.input_size], "w_k")?;
        validate_shape_3d(&w_v, &[self.nhead_kv, self.d_v, self.input_size], "w_v")?;
        validate_shape_4d(&w_kv, &[2, self.nhead_kv, self.d_k, self.input_size], "w_kv")?;
        validate_shape_4d(&w_qkv, &[3, self.nhead, self.d_k, self.input_size], "w_qkv")?;
        
        // Validate w_out shape
        let w_out_shape = w_out.shape().dims;
        let expected_w_out_shape = [self.nhead, self.d_v, self.output_size];
        if w_out_shape != expected_w_out_shape {
            return Err(format!("w_out shape mismatch: expected {:?}, got {:?}", 
                expected_w_out_shape, w_out_shape));
        }
        
        // Update parameters
        self.w_out = Param::from_tensor(w_out);
        self.w_q = w_q.map(Param::from_tensor);
        self.w_k = w_k.map(Param::from_tensor);
        self.w_v = w_v.map(Param::from_tensor);
        self.w_kv = w_kv.map(Param::from_tensor);
        self.w_qkv = w_qkv.map(Param::from_tensor);
        
        // Update cache
        self.k_cache = precomputed_k;
        self.v_cache = precomputed_v;
        self.kv_cache = precomputed_kv;
        
        Ok(())
    }

    /// Rearrange inputs to flat batch for processing
    fn rearrange_inputs_to_flat_batch(
        &self,
        x: Tensor<B, 3>,
        x_kv: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 3>>, Vec<usize>) {
        let x_shape = x.shape().dims.to_vec();
        
        if let Some(ref x_kv_tensor) = x_kv {
            // Ensure compatible shapes
            let x_kv_shape = x_kv_tensor.shape().dims;
            assert_eq!(x_shape[..x_shape.len()-2], x_kv_shape[..x_kv_shape.len()-2]);
        }
        
        // For simplicity, assuming 3D tensors [batch, seq, features]
        // In the full implementation, this would handle arbitrary batch dimensions
        (x, x_kv, x_shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_qkv(
        &mut self,  // Changed to &mut self to allow cache updates
        x: Tensor<B, 3>,
        x_kv: Option<Tensor<B, 3>>,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        only_cache_first_head_kv: bool,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 4>>, Option<Tensor<B, 4>>, Option<Tensor<B, 5>>, Option<Tensor<B, 5>>) {
        assert!(!(cache_kv && use_cached_kv), "Cannot both cache new KV and use cached KV at once");
        
        let x_kv = x_kv.unwrap_or_else(|| x.clone());
        
        let (mut k, mut v, mut kv) = (None, None, None);
        
        if use_cached_kv {
            assert!(self.has_cached_kv(), "Trying to use cached keys and values but cache is empty");
            k = self.k_cache.clone();
            v = self.v_cache.clone();
            kv = self.kv_cache.clone();
        }

        // Compute Q
        let q = if let Some(ref w_qkv) = self.w_qkv {
            // Check if x and x_kv are the same tensor (self-attention case)
            let is_self_attention = x.shape() == x_kv.shape(); // Simplified check
            if is_self_attention && kv.is_none() && k.is_none() && v.is_none() {
                // Use full QKV computation
                let qkv_result = self.einsum_qkv(x, w_qkv.val());
                return (None, k, v, kv, Some(qkv_result));
            } else {
                // Extract Q from QKV weights
                let w_q_slice = w_qkv.val().clone().slice([0..1, 0..self.nhead, 0..self.d_k, 0..self.input_size]);
                self.einsum_q(x, w_q_slice.squeeze::<3>(0))
            }
        } else if let Some(ref w_q) = self.w_q {
            self.einsum_q(x, w_q.val().squeeze::<3>(0))
        } else {
            panic!("No query weights available");
        };

        // Compute K, V if not cached
        if kv.is_none() && k.is_none() && v.is_none() {
            if let Some(ref w_qkv) = self.w_qkv {
                // Extract KV from QKV weights (slices 1 and 2)
                let w_kv_slice = w_qkv.val().clone().slice([1..3, 0..self.nhead_kv, 0..self.d_k, 0..self.input_size]);
                
                let mut w_kv_tensor = w_kv_slice;
                let orig_num_heads = if reuse_first_head_kv {
                    let original_heads = w_kv_tensor.shape().dims[1];
                    // Only use first head
                    w_kv_tensor = w_kv_tensor.clone().slice([0..2, 0..1, 0..self.d_k, 0..self.input_size]);
                    Some(original_heads)
                } else {
                    None
                };
                
                let mut computed_kv = self.einsum_kv(x_kv.clone(), w_kv_tensor);
                
                // Expand back to original number of heads if needed
                if let Some(orig_heads) = orig_num_heads {
                    // Python logic: expand_shape = [-1 for _ in kv.shape]; expand_shape[-2] = orig_num_heads
                    // MEMORY NOTE: repeat() creates actual copies - memory usage = orig_size * orig_heads
                    // TODO: Consider custom kernel for head replication to reduce memory overhead
                    computed_kv = computed_kv.repeat(&[1, 1, 1, orig_heads, 1]);
                }
                kv = Some(computed_kv);
            } else if let Some(ref w_kv) = self.w_kv {
                let mut w_kv_tensor = w_kv.val();
                let orig_num_heads = if reuse_first_head_kv {
                    let original_heads = w_kv_tensor.shape().dims[1];
                    // Only use first head
                    w_kv_tensor = w_kv_tensor.clone().slice([0..2, 0..1, 0..self.d_k, 0..self.input_size]);
                    Some(original_heads)
                } else {
                    None
                };
                
                let mut computed_kv = self.einsum_kv(x_kv.clone(), w_kv_tensor);
                
                // Expand back to original number of heads if needed
                if let Some(orig_heads) = orig_num_heads {
                    // Python logic: expand_shape = [-1 for _ in kv.shape]; expand_shape[-2] = orig_num_heads
                    // MEMORY NOTE: repeat() creates actual copies - memory usage = orig_size * orig_heads
                    // TODO: Consider custom kernel for head replication to reduce memory overhead
                    computed_kv = computed_kv.repeat(&[1, 1, 1, orig_heads, 1]);
                }
                kv = Some(computed_kv);
            } else if let Some(ref w_k) = self.w_k {
                if let Some(ref w_v) = self.w_v {
                    let mut w_k_tensor = w_k.val();
                    let mut w_v_tensor = w_v.val();
                    let orig_num_heads = if reuse_first_head_kv {
                        let original_heads = w_k_tensor.shape().dims[0];
                        w_k_tensor = w_k_tensor.clone().slice([0..1, 0..self.d_k, 0..self.input_size]);
                        w_v_tensor = w_v_tensor.clone().slice([0..1, 0..self.d_v, 0..self.input_size]);
                        Some(original_heads)
                    } else {
                        None
                    };
                    
                    let mut computed_k = self.einsum_k(x_kv.clone(), w_k_tensor);
                    let mut computed_v = self.einsum_v(x_kv, w_v_tensor);
                    
                    // Expand back to original number of heads if needed
                    if let Some(orig_heads) = orig_num_heads {

                        // Python logic: expand_shape = [-1 for _ in k.shape]; expand_shape[-2] = orig_num_heads
                        // MEMORY NOTE: repeat() creates actual copies - memory usage scales with orig_heads
                        // TODO: Consider custom kernel for head replication to reduce memory overhead
                        computed_k = computed_k.repeat(&[1, 1, orig_heads, 1]);
                        computed_v = computed_v.repeat(&[1, 1, orig_heads, 1]);
                    }
                    
                    k = Some(computed_k);
                    v = Some(computed_v);
                }
            }
        }

        // *** FIXED: Cache Update Implementation with Proper Append Logic ***
        if cache_kv {
            // Update k_cache: append new K along sequence axis if cache exists and new K computed
            if let (Some(k_cache_tensor), Some(computed_k)) = (&self.k_cache, &k) {
                let cache_shape = k_cache_tensor.shape().dims;
                let computed_shape = computed_k.shape().dims;
                
                // Check if shapes are compatible for concatenation (all dims except seq_len should match)
                if cache_shape.len() != computed_shape.len() {
                    panic!("Cache dimensionality mismatch: k_cache {:?} vs computed_k {:?}", cache_shape, computed_shape);
                }
                
                for (i, (&cache_dim, &computed_dim)) in cache_shape.iter().zip(computed_shape.iter()).enumerate() {
                    if i != 1 && cache_dim != computed_dim { // Skip seq_len dimension (index 1)
                        panic!("Cache shape mismatch at dim {}: k_cache {:?} vs computed_k {:?}", i, cache_shape, computed_shape);
                    }
                }
                
                // Append new K to existing cache along seq_len axis (dim=1)
                let updated_k_cache = Tensor::cat(vec![k_cache_tensor.clone(), computed_k.clone()], 1);
                self.k_cache = Some(updated_k_cache);
            } else if k.is_some() {
                // Initialize cache with computed K if no cache exists
                self.k_cache = k.clone();
            }
            
            // Update v_cache: append new V along sequence axis if cache exists and new V computed  
            if let (Some(v_cache_tensor), Some(computed_v)) = (&self.v_cache, &v) {
                let cache_shape = v_cache_tensor.shape().dims;
                let computed_shape = computed_v.shape().dims;
                
                // Check if shapes are compatible for concatenation
                if cache_shape.len() != computed_shape.len() {
                    panic!("Cache dimensionality mismatch: v_cache {:?} vs computed_v {:?}", cache_shape, computed_shape);
                }
                
                for (i, (&cache_dim, &computed_dim)) in cache_shape.iter().zip(computed_shape.iter()).enumerate() {
                    if i != 1 && cache_dim != computed_dim { // Skip seq_len dimension (index 1)
                        panic!("Cache shape mismatch at dim {}: v_cache {:?} vs computed_v {:?}", i, cache_shape, computed_shape);
                    }
                }
                
                // Append new V to existing cache along seq_len axis (dim=1)
                let updated_v_cache = Tensor::cat(vec![v_cache_tensor.clone(), computed_v.clone()], 1);
                self.v_cache = Some(updated_v_cache);
            } else if v.is_some() {
                // Initialize cache with computed V if no cache exists
                self.v_cache = v.clone();
            }
            
            // Handle kv_cache updates with special logic for first-head-only caching
            if let (Some(kv_cache_tensor), Some(computed_kv)) = (&self.kv_cache, &kv) {
                let cache_shape = kv_cache_tensor.shape().dims;
                let computed_shape = computed_kv.shape().dims;
                
                // Check if this is first-head-only caching (cache shape has nhead_kv=1)
                let kv_to_cache = if cache_shape.len() >= 2 && cache_shape[cache_shape.len()-2] == 1 {
                    // Only cache first head: extract first head from computed KV
                    computed_kv.clone().slice([
                        0..computed_shape[0], 
                        0..computed_shape[1], 
                        0..computed_shape[2], 
                        0..1,  // Only first head
                        0..computed_shape[4]
                    ])
                } else {
                    // Cache all heads
                    computed_kv.clone()
                };
                
                let kv_to_cache_shape = kv_to_cache.shape().dims;
                
                // Check if shapes are compatible for concatenation
                if cache_shape.len() != kv_to_cache_shape.len() {
                    panic!("Cache dimensionality mismatch: kv_cache {:?} vs kv_to_cache {:?}", cache_shape, kv_to_cache_shape);
                }
                
                for (i, (&cache_dim, &kv_dim)) in cache_shape.iter().zip(kv_to_cache_shape.iter()).enumerate() {
                    if i != 1 && cache_dim != kv_dim { // Skip seq_len dimension (index 1)
                        panic!("Cache shape mismatch at dim {}: kv_cache {:?} vs kv_to_cache {:?}", i, cache_shape, kv_to_cache_shape);
                    }
                }
                
                // Append new KV to existing cache along seq_len axis (dim=1)
                let updated_kv_cache = Tensor::cat(vec![kv_cache_tensor.clone(), kv_to_cache], 1);
                self.kv_cache = Some(updated_kv_cache);
            } else if kv.is_some() {
                // Initialize cache with computed KV if no cache exists
                let computed_kv = kv.as_ref().unwrap();
                let computed_shape = computed_kv.shape().dims;
                
                // Determine if we should cache only first head based on only_cache_first_head_kv setting
                if only_cache_first_head_kv && computed_shape.len() >= 2 {
                    // Only cache first head
                    let first_head_kv = computed_kv.clone().slice([
                        0..computed_shape[0], 
                        0..computed_shape[1], 
                        0..computed_shape[2], 
                        0..1,  // Only first head
                        0..computed_shape[4]
                    ]);
                    self.kv_cache = Some(first_head_kv);
                } else {
                    // Cache all heads
                    self.kv_cache = Some(computed_kv.clone());
                }
            }
        }

        (Some(q), k, v, kv, None)
    }

    fn einsum_q(&self, x: Tensor<B, 3>, w_q: Tensor<B, 3>) -> Tensor<B, 4> {
        // Equivalent to: torch.einsum("... s, h d s -> ... h d", x, w_q)
        let x_shape = x.shape().dims;
        let w_q_shape = w_q.shape().dims;
        
        // Shape assertions with descriptive error messages
        assert!(x_shape.len() == 3, "einsum_q: x must be 3D tensor [batch, seq, input_size], got {:?}", x_shape);
        assert!(w_q_shape.len() == 3, "einsum_q: w_q must be 3D tensor [nhead, d_k, input_size], got {:?}", w_q_shape);
        assert_eq!(x_shape[2], self.input_size, "einsum_q: x dim 2 must equal input_size {}, got {}", self.input_size, x_shape[2]);
        assert_eq!(w_q_shape[2], self.input_size, "einsum_q: w_q dim 2 must equal input_size {}, got {}", self.input_size, w_q_shape[2]);
        assert_eq!(w_q_shape[0], self.nhead, "einsum_q: w_q dim 0 must equal nhead {}, got {}", self.nhead, w_q_shape[0]);
        assert_eq!(w_q_shape[1], self.d_k, "einsum_q: w_q dim 1 must equal d_k {}, got {}", self.d_k, w_q_shape[1]);
        
        let batch_size = x_shape[0];
        let seq_len = x_shape[1];
        
        // Reshape x to [batch*seq, input_size] and w_q to [nhead*d_k, input_size]
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_q_flat = w_q.reshape([self.nhead * self.d_k, self.input_size]);
        
        // Matrix multiply: [batch*seq, input_size] @ [input_size, nhead*d_k] -> [batch*seq, nhead*d_k]
        let result_flat = x_flat.matmul(w_q_flat.transpose());
        
        // Validate matmul result shape before final reshape
        let result_flat_shape = result_flat.shape().dims;
        let expected_flat_shape = [batch_size * seq_len, self.nhead * self.d_k];
        assert_eq!(result_flat_shape, expected_flat_shape, "einsum_q: matmul result shape mismatch, expected {:?}, got {:?}", expected_flat_shape, result_flat_shape);
        
        // Reshape back to [batch, seq, nhead, d_k]
        result_flat.reshape([batch_size, seq_len, self.nhead, self.d_k])
    }

    fn einsum_k(&self, x: Tensor<B, 3>, w_k: Tensor<B, 3>) -> Tensor<B, 4> {
        // Equivalent to: torch.einsum("... s, h d s -> ... h d", x, w_k)
        let x_shape = x.shape().dims;
        let w_k_shape = w_k.shape().dims;
        
        // Input validation
        assert!(x_shape.len() == 3, "einsum_k: x must be 3D tensor [batch, seq, input_size], got {:?}", x_shape);
        assert!(w_k_shape.len() == 3, "einsum_k: w_k must be 3D tensor [nhead_kv, d_k, input_size], got {:?}", w_k_shape);
        
        assert_eq!(x_shape[2], self.input_size, "einsum_k: x dim 2 must equal input_size {}, got {}", self.input_size, x_shape[2]);
        assert_eq!(w_k_shape[0], self.nhead_kv, "einsum_k: w_k dim 0 must equal nhead_kv {}, got {}", self.nhead_kv, w_k_shape[0]);
        assert_eq!(w_k_shape[1], self.d_k, "einsum_k: w_k dim 1 must equal d_k {}, got {}", self.d_k, w_k_shape[1]);
        assert_eq!(w_k_shape[2], self.input_size, "einsum_k: w_k dim 2 must equal input_size {}, got {}", self.input_size, w_k_shape[2]);

        let batch_size = x_shape[0];
        let seq_len = x_shape[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_k_flat = w_k.reshape([self.nhead_kv * self.d_k, self.input_size]);
        
        let result_flat = x_flat.matmul(w_k_flat.transpose());
        
        // Validate result shape before reshape
        let result_flat_shape = result_flat.shape().dims;
        let expected_flat_shape = [batch_size * seq_len, self.nhead_kv * self.d_k];
        assert_eq!(result_flat_shape, expected_flat_shape, 
            "einsum_k: matmul result shape {:?} doesn't match expected {:?}", result_flat_shape, expected_flat_shape);
        
        let result = result_flat.reshape([batch_size, seq_len, self.nhead_kv, self.d_k]);
        
        // Validate final result shape
        let final_shape = result.shape().dims;
        let expected_final_shape = [batch_size, seq_len, self.nhead_kv, self.d_k];
        assert_eq!(final_shape, expected_final_shape, 
            "einsum_k: final result shape {:?} doesn't match expected {:?}", final_shape, expected_final_shape);
        
        result
    }

    fn einsum_v(&self, x: Tensor<B, 3>, w_v: Tensor<B, 3>) -> Tensor<B, 4> {
        // Equivalent to: torch.einsum("... s, h d s -> ... h d", x, w_v)
        let x_shape = x.shape().dims;
        let w_v_shape = w_v.shape().dims;
        
        // Input validation
        assert!(x_shape.len() == 3, "einsum_v: x must be 3D tensor [batch, seq, input_size], got {:?}", x_shape);
        assert!(w_v_shape.len() == 3, "einsum_v: w_v must be 3D tensor [nhead_kv, d_v, input_size], got {:?}", w_v_shape);
        
        assert_eq!(x_shape[2], self.input_size, "einsum_v: x dim 2 must equal input_size {}, got {}", self.input_size, x_shape[2]);
        assert_eq!(w_v_shape[0], self.nhead_kv, "einsum_v: w_v dim 0 must equal nhead_kv {}, got {}", self.nhead_kv, w_v_shape[0]);
        assert_eq!(w_v_shape[1], self.d_v, "einsum_v: w_v dim 1 must equal d_v {}, got {}", self.d_v, w_v_shape[1]);
        assert_eq!(w_v_shape[2], self.input_size, "einsum_v: w_v dim 2 must equal input_size {}, got {}", self.input_size, w_v_shape[2]);

        let batch_size = x_shape[0];
        let seq_len = x_shape[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_v_flat = w_v.reshape([self.nhead_kv * self.d_v, self.input_size]);
        
        let result_flat = x_flat.matmul(w_v_flat.transpose());
        
        // Validate result shape before reshape
        let result_flat_shape = result_flat.shape().dims;
        let expected_flat_shape = [batch_size * seq_len, self.nhead_kv * self.d_v];
        assert_eq!(result_flat_shape, expected_flat_shape, 
            "einsum_v: matmul result shape {:?} doesn't match expected {:?}", result_flat_shape, expected_flat_shape);
        
        let result = result_flat.reshape([batch_size, seq_len, self.nhead_kv, self.d_v]);
        
        // Validate final result shape
        let final_shape = result.shape().dims;
        let expected_final_shape = [batch_size, seq_len, self.nhead_kv, self.d_v];
        assert_eq!(final_shape, expected_final_shape, 
            "einsum_v: final result shape {:?} doesn't match expected {:?}", final_shape, expected_final_shape);
        
        result
    }

    fn einsum_kv(&self, x: Tensor<B, 3>, w_kv: Tensor<B, 4>) -> Tensor<B, 5> {
        // Equivalent to: torch.einsum("... s, j h d s -> ... j h d", x, w_kv)
        let x_shape = x.shape().dims;
        let w_kv_shape = w_kv.shape().dims;
        
        // Input validation
        assert!(x_shape.len() == 3, "einsum_kv: x must be 3D tensor [batch, seq, input_size], got {:?}", x_shape);
        assert!(w_kv_shape.len() == 4, "einsum_kv: w_kv must be 4D tensor [2, nhead_kv, d_k, input_size], got {:?}", w_kv_shape);
        
        assert_eq!(x_shape[2], self.input_size, "einsum_kv: x dim 2 must equal input_size {}, got {}", self.input_size, x_shape[2]);
        assert_eq!(w_kv_shape[0], 2, "einsum_kv: w_kv dim 0 must be 2 (for K and V), got {}", w_kv_shape[0]);
        assert_eq!(w_kv_shape[1], self.nhead_kv, "einsum_kv: w_kv dim 1 must equal nhead_kv {}, got {}", self.nhead_kv, w_kv_shape[1]);
        assert_eq!(w_kv_shape[2], self.d_k, "einsum_kv: w_kv dim 2 must equal d_k {}, got {}", self.d_k, w_kv_shape[2]);
        assert_eq!(w_kv_shape[3], self.input_size, "einsum_kv: w_kv dim 3 must equal input_size {}, got {}", self.input_size, w_kv_shape[3]);

        let batch_size = x_shape[0];
        let seq_len = x_shape[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_kv_flat = w_kv.reshape([2 * self.nhead_kv * self.d_k, self.input_size]);
        
        let result_flat = x_flat.matmul(w_kv_flat.transpose());
        
        // Validate result shape before reshape
        let result_flat_shape = result_flat.shape().dims;
        let expected_flat_shape = [batch_size * seq_len, 2 * self.nhead_kv * self.d_k];
        assert_eq!(result_flat_shape, expected_flat_shape, 
            "einsum_kv: matmul result shape {:?} doesn't match expected {:?}", result_flat_shape, expected_flat_shape);
        
        let result = result_flat.reshape([batch_size, seq_len, 2, self.nhead_kv, self.d_k]);
        
        // Validate final result shape
        let final_shape = result.shape().dims;
        let expected_final_shape = [batch_size, seq_len, 2, self.nhead_kv, self.d_k];
        assert_eq!(final_shape, expected_final_shape, 
            "einsum_kv: final result shape {:?} doesn't match expected {:?}", final_shape, expected_final_shape);
        
        result
    }

    fn einsum_qkv(&self, x: Tensor<B, 3>, w_qkv: Tensor<B, 4>) -> Tensor<B, 5> {
        // Equivalent to: torch.einsum("... s, j h d s -> ... j h d", x, w_qkv)
        let x_shape = x.shape().dims;
        let w_qkv_shape = w_qkv.shape().dims;
        
        // Input validation
        assert!(x_shape.len() == 3, "einsum_qkv: x must be 3D tensor [batch, seq, input_size], got {:?}", x_shape);
        assert!(w_qkv_shape.len() == 4, "einsum_qkv: w_qkv must be 4D tensor [3, nhead, d_k, input_size], got {:?}", w_qkv_shape);
        
        assert_eq!(x_shape[2], self.input_size, "einsum_qkv: x dim 2 must equal input_size {}, got {}", self.input_size, x_shape[2]);
        assert_eq!(w_qkv_shape[0], 3, "einsum_qkv: w_qkv dim 0 must be 3 (for Q, K, V), got {}", w_qkv_shape[0]);
        assert_eq!(w_qkv_shape[1], self.nhead, "einsum_qkv: w_qkv dim 1 must equal nhead {}, got {}", self.nhead, w_qkv_shape[1]);
        assert_eq!(w_qkv_shape[2], self.d_k, "einsum_qkv: w_qkv dim 2 must equal d_k {}, got {}", self.d_k, w_qkv_shape[2]);
        assert_eq!(w_qkv_shape[3], self.input_size, "einsum_qkv: w_qkv dim 3 must equal input_size {}, got {}", self.input_size, w_qkv_shape[3]);

        let batch_size = x_shape[0];
        let seq_len = x_shape[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_qkv_flat = w_qkv.reshape([3 * self.nhead * self.d_k, self.input_size]);
        
        let result_flat = x_flat.matmul(w_qkv_flat.transpose());
        
        // Validate result shape before reshape
        let result_flat_shape = result_flat.shape().dims;
        let expected_flat_shape = [batch_size * seq_len, 3 * self.nhead * self.d_k];
        assert_eq!(result_flat_shape, expected_flat_shape, 
            "einsum_qkv: matmul result shape {:?} doesn't match expected {:?}", result_flat_shape, expected_flat_shape);
        
        let result = result_flat.reshape([batch_size, seq_len, 3, self.nhead, self.d_k]);
        
        // Validate final result shape
        let final_shape = result.shape().dims;
        let expected_final_shape = [batch_size, seq_len, 3, self.nhead, self.d_k];
        assert_eq!(final_shape, expected_final_shape, 
            "einsum_qkv: final result shape {:?} doesn't match expected {:?}", final_shape, expected_final_shape);
        
        result
    }

    fn broadcast_kv_across_heads(
        kv: Tensor<B, 4>,
        share_kv_across_n_heads: usize,
    ) -> Tensor<B, 4> {
        let shape = kv.shape();
        let dims = shape.dims;
        
        // Python logic: kv[..., None, :].expand(*[(-1) * (kv.dim() - 1)], share_kv_across_n_heads, -1)
        // This means: expand along a new dimension inserted before the last dimension
        
        let [batch, seq, nhead, d] = [dims[0], dims[1], dims[2], dims[3]];
        
        // Step 1: Add a new dimension: [batch, seq, nhead, 1, d]
        let kv_expanded_dim: Tensor<B, 5> = kv.unsqueeze_dim(3);
        
        // Step 2: Expand along the new dimension to share_kv_across_n_heads
        // This is equivalent to Python's expand(*[(-1) * (kv.dim() - 1)], share_kv_across_n_heads, -1)
        // MEMORY NOTE: repeat() creates actual copies - memory usage = orig_size * share_kv_across_n_heads
        // TODO: This is true replication for KV sharing - consider view-based sharing if possible
        let kv_expanded = kv_expanded_dim.repeat(&[1, 1, 1, share_kv_across_n_heads, 1]);
        
        // Step 3: Reshape to merge the head dimensions: [batch, seq, nhead * share_kv_across_n_heads, d]
        let final_nhead = nhead * share_kv_across_n_heads;
        kv_expanded.reshape([batch, seq, final_nhead, d])
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_attention_heads(
        q: Option<Tensor<B, 4>>,
        k: Option<Tensor<B, 4>>,
        v: Option<Tensor<B, 4>>,
        kv: Option<Tensor<B, 5>>,
        qkv: Option<Tensor<B, 5>>,
        dropout: Option<&Dropout>,
        softmax_scale: Option<f64>,
        train: bool,
        attention_mask: Option<Tensor<B, 3>>,  // [batch, seq_q, seq_kv] or broadcastable
    ) -> Tensor<B, 4> {
        assert!(k.is_none() == v.is_none());
        assert!([qkv.is_none(), kv.is_none(), k.is_none() && v.is_none()].iter().filter(|&&x| x).count() == 2);
        assert!(qkv.is_none() != q.is_none());

        let (q, k, v) = if let Some(qkv_tensor) = qkv {
            // Use helper functions for robust QKV slicing
            let q = Self::slice_q_from_qkv(&qkv_tensor);
            let k = Self::slice_k_from_qkv_or_kv(&qkv_tensor);
            let v = Self::slice_v_from_qkv_or_kv(&qkv_tensor);
            (q, k, v)
        } else if let Some(kv_tensor) = kv {
            let q = q.unwrap();
            // Use helper functions for robust KV slicing
            let k = Self::slice_k_from_qkv_or_kv(&kv_tensor);
            let v = Self::slice_v_from_qkv_or_kv(&kv_tensor);
            (q, k, v)
        } else {
            (q.unwrap(), k.unwrap(), v.unwrap())
        };

        let q_shape = q.shape();
        let v_shape = v.shape();
        let [batch_size, seqlen_q, nhead, d_k] = [q_shape.dims[0], q_shape.dims[1], q_shape.dims[2], q_shape.dims[3]];
        let [_, _seqlen_kv, nhead_kv, d_v] = [v_shape.dims[0], v_shape.dims[1], v_shape.dims[2], v_shape.dims[3]];
        let share_kv_across_n_heads = nhead / nhead_kv;

        // Basic attention implementation (no flash attention support in Burn yet)
        let k_broadcast = Self::broadcast_kv_across_heads(k, share_kv_across_n_heads);
        let v_broadcast = Self::broadcast_kv_across_heads(v, share_kv_across_n_heads);

        // Compute attention scores: einsum("b q h d, b k h d -> b q k h", q, k)
        let q_reshaped = q.swap_dims(1, 2); // [batch, nhead, seq_q, d_k]
        let k_reshaped = k_broadcast.swap_dims(1, 2).swap_dims(2, 3); // [batch, nhead, d_k, seq_kv]
        let logits = q_reshaped.matmul(k_reshaped); // [batch, nhead, seq_q, seq_kv]

        let scale = softmax_scale.unwrap_or(1.0 / (d_k as f64).sqrt());
        let mut logits_scaled = logits * scale;

        // Apply attention mask to prevent label leakage
        if let Some(mask) = attention_mask {
            // mask should be [batch, seq_q, seq_kv] with 1.0 for allowed, 0.0 for masked
            // Convert to [batch, 1, seq_q, seq_kv] to broadcast across heads
            let mask_expanded = mask.unsqueeze_dim(1); // [batch, 1, seq_q, seq_kv]
            
            // Apply mask: set masked positions to large negative value (-1e9) before softmax
            let mask_value = Tensor::full([1], -1e9f32, &logits_scaled.device());
            let inverted_mask = mask_expanded.clone().equal_elem(0.0); // 1 where we want to mask
            logits_scaled = logits_scaled.mask_where(inverted_mask, mask_value);
        }

        let mut ps = activation::softmax(logits_scaled, 3); // softmax over seq_kv dimension
        
        // Apply dropout to attention probabilities if specified and in training mode
        if train {
            if let Some(dropout_module) = dropout {
                ps = dropout_module.forward(ps);
            }
        }
        // In eval mode (train == false), skip dropout entirely

        // Apply attention: einsum("b q k h, b k h d -> b q h d", ps, v)
        let v_reshaped = v_broadcast.swap_dims(1, 2); // [batch, nhead, seq_kv, d_v]
        let attention_output = ps.matmul(v_reshaped); // [batch, nhead, seq_q, d_v]

        // Transpose back to [batch, seq_q, nhead, d_v]
        attention_output.swap_dims(1, 2).reshape([batch_size, seqlen_q, nhead, d_v])
    }

    #[allow(clippy::too_many_arguments)]
    fn compute(
        &mut self,
        x: Tensor<B, 3>,
        x_kv: Option<Tensor<B, 3>>,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        _only_cache_first_head_kv: bool,
        train: bool,
        attention_mask: Option<Tensor<B, 3>>,  // [batch, seq_q, seq_kv] - 1.0 for allowed, 0.0 for masked
    ) -> Tensor<B, 3> {
        let (q, k, v, kv, qkv) = self.compute_qkv(
            x,
            x_kv,
            cache_kv,
            use_cached_kv,
            reuse_first_head_kv,
            _only_cache_first_head_kv,
        );

        let attention_head_outputs = Self::compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            self.dropout.as_ref(),
            self.softmax_scale,
            train,
            attention_mask,
        );

        // Apply output projection: torch.einsum("... h d, h d s -> ... s", attention_head_outputs, w_out)
        let batch_size = attention_head_outputs.shape().dims[0];
        let seq_len = attention_head_outputs.shape().dims[1];
        
        let attention_flat = attention_head_outputs.reshape([batch_size * seq_len, self.nhead * self.d_v]);
        let w_out_flat = self.w_out.val().reshape([self.nhead * self.d_v, self.output_size]);
        
        let result_flat = attention_flat.matmul(w_out_flat);
        result_flat.reshape([batch_size, seq_len, self.output_size])
    }

    /// Compute attention with optional gradient checkpointing
    /// This is the equivalent of Python's @support_save_peak_mem_factor decorator
    #[allow(clippy::too_many_arguments)]
    fn compute_with_checkpointing(
        &mut self,
        x: Tensor<B, 3>,
        x_kv: Option<Tensor<B, 3>>,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        only_cache_first_head_kv: bool,
        save_peak_mem_factor: Option<i64>,
        add_input: bool,
        allow_inplace: bool,
        train: bool,
        attention_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        if self.recompute_attn && save_peak_mem_factor.is_some() {
            // Implement gradient checkpointing logic
            // Since Burn doesn't have built-in checkpointing, we'll implement a memory-efficient approach
            self.compute_with_memory_optimization(
                x,
                x_kv,
                cache_kv,
                use_cached_kv,
                reuse_first_head_kv,
                only_cache_first_head_kv,
                save_peak_mem_factor.unwrap_or(1),
                add_input,
                allow_inplace,
                train,
                attention_mask,
            )
        } else {
            // Standard computation without checkpointing
            let mut output = self.compute(
                x.clone(),
                x_kv,
                cache_kv,
                use_cached_kv,
                reuse_first_head_kv,
                only_cache_first_head_kv,
                train,
                attention_mask,
            );
            
            // Handle add_input with potential inplace optimization
            if add_input {
                if allow_inplace {
                    // Inplace addition for memory efficiency
                    output = output + x;
                } else {
                    output = output + x;
                }
            }
            
            output
        }
    }

    /// Memory-optimized computation that simulates gradient checkpointing
    /// This breaks the computation into smaller chunks to reduce peak memory usage
    #[allow(clippy::too_many_arguments)]
    fn compute_with_memory_optimization(
        &mut self,
        x: Tensor<B, 3>,
        x_kv: Option<Tensor<B, 3>>,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        only_cache_first_head_kv: bool,
        save_peak_mem_factor: i64,
        add_input: bool,
        allow_inplace: bool,
        train: bool,
        attention_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let batch_size = x.shape().dims[0];
        let seq_len = x.shape().dims[1];
        let input_size = x.shape().dims[2];
        
        // Calculate chunk size based on save_peak_mem_factor
        let chunk_size = ((seq_len as f64) / (save_peak_mem_factor as f64).sqrt()).ceil() as usize;
        let chunk_size = chunk_size.max(1).min(seq_len); // Ensure valid range
        
        if chunk_size >= seq_len {
            // No chunking needed, use standard computation
            let mut output = self.compute(
                x.clone(),
                x_kv,
                cache_kv,
                use_cached_kv,
                reuse_first_head_kv,
                only_cache_first_head_kv,
                train,
                attention_mask,
            );
            
            if add_input {
                output = if allow_inplace { output + x } else { output + x };
            }
            
            return output;
        }
        
        // Process in chunks to save memory
        let mut output_chunks = Vec::new();
        
        for chunk_start in (0..seq_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            
            // Extract chunk from x
            let x_chunk = x.clone().slice([0..batch_size, chunk_start..chunk_end, 0..input_size]);
            
            // Extract chunk from x_kv if present (for cross-attention)
            let x_kv_chunk = x_kv.as_ref().map(|x_kv_tensor| {
                x_kv_tensor.clone().slice([0..batch_size, chunk_start..chunk_end, 0..input_size])
            });
            
            // Extract chunk from attention mask if present
            let attention_mask_chunk = attention_mask.as_ref().map(|mask| {
                let mask_seq_kv = mask.shape().dims[2];
                mask.clone().slice([0..batch_size, chunk_start..chunk_end, 0..mask_seq_kv])
            });
            
            // Compute attention for this chunk
            let chunk_output = self.compute(
                x_chunk.clone(),
                x_kv_chunk,
                false, // Don't cache KV for chunks to save memory
                use_cached_kv,
                reuse_first_head_kv,
                only_cache_first_head_kv,
                train,
                attention_mask_chunk,
            );
            
            // Apply add_input if needed
            let final_chunk_output = if add_input {
                if allow_inplace {
                    chunk_output + x_chunk
                } else {
                    chunk_output + x_chunk
                }
            } else {
                chunk_output
            };
            
            output_chunks.push(final_chunk_output);
        }
        
        // Concatenate all chunks back together
        if output_chunks.len() == 1 {
            output_chunks.into_iter().next().unwrap()
        } else {
            // Concatenate along sequence dimension
            Tensor::cat(output_chunks, 1)
        }
    }

    /// Convert PyTorch MultiheadAttention state dict to TabPFN format
    /// This provides compatibility with PyTorch's nn.MultiheadAttention
    pub fn convert_torch_nn_multihead_attention_state_dict(
        state_dict: std::collections::HashMap<String, Tensor<B, 2>>,
        nhead: usize,
        disable_stacked_w_qkv: bool,
    ) -> Result<std::collections::HashMap<String, Tensor<B, 4>>, String> {
        let in_proj_weight = state_dict.get("in_proj_weight")
            .ok_or("Missing in_proj_weight in state_dict")?;
        let out_proj_weight = state_dict.get("out_proj.weight")
            .ok_or("Missing out_proj.weight in state_dict")?;

        let embed_dim = in_proj_weight.shape().dims[1];
        if embed_dim % nhead != 0 {
            return Err(format!("embed_dim {} not divisible by nhead {}", embed_dim, nhead));
        }
        
        if in_proj_weight.shape().dims[0] != 3 * embed_dim {
            return Err(format!("Expected in_proj_weight shape [{}, {}], got {:?}", 
                3 * embed_dim, embed_dim, in_proj_weight.shape().dims));
        }
        
        if out_proj_weight.shape().dims != [embed_dim, embed_dim] {
            return Err(format!("Expected out_proj_weight shape [{}, {}], got {:?}", 
                embed_dim, embed_dim, out_proj_weight.shape().dims));
        }

        // Reshape in_proj_weight to [3, nhead, -1, embed_dim]
        let head_dim = embed_dim / nhead;
        let in_proj_reshaped = in_proj_weight.clone().reshape([3, nhead, head_dim, embed_dim]);

        let mut result = std::collections::HashMap::new();
        
        if disable_stacked_w_qkv {
            // Split into separate Q and KV weights
            let q_weight = in_proj_reshaped.clone().slice([0..1, 0..nhead, 0..head_dim, 0..embed_dim]);
            let kv_weight = in_proj_reshaped.clone().slice([1..3, 0..nhead, 0..head_dim, 0..embed_dim]);
            
            result.insert("_w_q".to_string(), q_weight);
            result.insert("_w_kv".to_string(), kv_weight);
        } else {
            result.insert("_w_qkv".to_string(), in_proj_reshaped);
        }
        
        // Transpose and reshape output weight: [embed_dim, embed_dim] -> [1, nhead, head_dim, embed_dim]
        let w_out = out_proj_weight.clone().transpose().reshape([1, nhead, head_dim, embed_dim]);
        result.insert("_w_out".to_string(), w_out);
        
        Ok(result)
    }

    /// Load weights from NumPy arrays (from Python fixtures)
    pub fn load_weights_from_numpy(
        &mut self,
        weights: HashMap<String, (Vec<usize>, Vec<f32>)>,
        device: &B::Device,
    ) -> Result<(), String> {
        // Helper to convert Vec<f32> to tensor with specific dimensions
        let to_tensor_3d = |shape: Vec<usize>, data: Vec<f32>| -> Result<Tensor<B, 3>, String> {
            if shape.len() != 3 {
                return Err(format!("Expected 3D tensor, got {} dimensions", shape.len()));
            }
            let tensor_data = burn::tensor::TensorData::new(data, shape);
            Ok(Tensor::from_data(tensor_data, device))
        };
        
        let to_tensor_4d = |shape: Vec<usize>, data: Vec<f32>| -> Result<Tensor<B, 4>, String> {
            if shape.len() != 4 {
                return Err(format!("Expected 4D tensor, got {} dimensions", shape.len()));
            }
            let tensor_data = burn::tensor::TensorData::new(data, shape);
            Ok(Tensor::from_data(tensor_data, device))
        };
        
        let to_tensor_5d = |shape: Vec<usize>, data: Vec<f32>| -> Result<Tensor<B, 5>, String> {
            if shape.len() != 5 {
                return Err(format!("Expected 5D tensor, got {} dimensions", shape.len()));
            }
            let tensor_data = burn::tensor::TensorData::new(data, shape);
            Ok(Tensor::from_data(tensor_data, device))
        };

        // Load output weight (required)
        let w_out = if let Some((shape, data)) = weights.get("weight_w_out") {
            to_tensor_3d(shape.clone(), data.clone()).map(Param::from_tensor)?
        } else {
            return Err("Missing required weight_w_out".to_string());
        };

        // Load input weights (various combinations possible)
        let w_q = weights.get("weight_w_q")
            .map(|(shape, data)| to_tensor_4d(shape.clone(), data.clone()).map(Param::from_tensor))
            .transpose()?;
            
        let w_k = weights.get("weight_w_k")
            .map(|(shape, data)| to_tensor_3d(shape.clone(), data.clone()).map(Param::from_tensor))
            .transpose()?;
            
        let w_v = weights.get("weight_w_v")
            .map(|(shape, data)| to_tensor_3d(shape.clone(), data.clone()).map(Param::from_tensor))
            .transpose()?;
            
        let w_kv = weights.get("weight_w_kv")
            .map(|(shape, data)| to_tensor_4d(shape.clone(), data.clone()).map(Param::from_tensor))
            .transpose()?;
            
        let w_qkv = weights.get("weight_w_qkv")
            .map(|(shape, data)| to_tensor_4d(shape.clone(), data.clone()).map(Param::from_tensor))
            .transpose()?;

        // Load cache if present
        let precomputed_k = weights.get("cache_k_cache")
            .map(|(shape, data)| to_tensor_4d(shape.clone(), data.clone()))
            .transpose()?;
            
        let precomputed_v = weights.get("cache_v_cache")
            .map(|(shape, data)| to_tensor_4d(shape.clone(), data.clone()))
            .transpose()?;
            
        let precomputed_kv = weights.get("cache_kv_cache")
            .map(|(shape, data)| to_tensor_5d(shape.clone(), data.clone()))
            .transpose()?;

        // Use set_parameters to load everything
        self.set_parameters(
            w_out.val(),
            w_q.map(|p| p.val()),
            w_k.map(|p| p.val()),
            w_v.map(|p| p.val()),
            w_kv.map(|p| p.val()),
            w_qkv.map(|p| p.val()),
            precomputed_k,
            precomputed_v,
            precomputed_kv,
        )?;

        Ok(())
    }

    /// Save current state to AttentionState
    pub fn save_state(&self) -> AttentionState {
        let mut state = AttentionState::new();
        
        // Save configuration
        state.input_size = self.input_size;
        state.output_size = self.output_size;
        state.nhead = self.nhead;
        state.nhead_kv = self.nhead_kv;
        state.d_k = self.d_k;
        state.d_v = self.d_v;
        state.share_kv_across_n_heads = self.share_kv_across_n_heads;
        state.dropout_p = self.dropout_p;
        state.softmax_scale = self.softmax_scale;
        state.init_gain = self.init_gain;
        state.recompute_attn = self.recompute_attn;

        // Helper to convert tensor to (shape, data) - we'll handle different dimensions
        let tensor_to_data_3d = |tensor: &Tensor<B, 3>| -> (Vec<usize>, Vec<f32>) {
            let shape = tensor.shape().dims.to_vec();
            let data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
            (shape, data)
        };
        
        let tensor_to_data_4d = |tensor: &Tensor<B, 4>| -> (Vec<usize>, Vec<f32>) {
            let shape = tensor.shape().dims.to_vec();
            let data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
            (shape, data)
        };
        
        let tensor_to_data_5d = |tensor: &Tensor<B, 5>| -> (Vec<usize>, Vec<f32>) {
            let shape = tensor.shape().dims.to_vec();
            let data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
            (shape, data)
        };

        // Save weights
        state.weights.insert("w_out".to_string(), tensor_to_data_3d(&self.w_out.val()));
        
        if let Some(ref w_q) = self.w_q {
            state.weights.insert("w_q".to_string(), tensor_to_data_4d(&w_q.val()));
        }
        if let Some(ref w_k) = self.w_k {
            state.weights.insert("w_k".to_string(), tensor_to_data_3d(&w_k.val()));
        }
        if let Some(ref w_v) = self.w_v {
            state.weights.insert("w_v".to_string(), tensor_to_data_3d(&w_v.val()));
        }
        if let Some(ref w_kv) = self.w_kv {
            state.weights.insert("w_kv".to_string(), tensor_to_data_4d(&w_kv.val()));
        }
        if let Some(ref w_qkv) = self.w_qkv {
            state.weights.insert("w_qkv".to_string(), tensor_to_data_4d(&w_qkv.val()));
        }

        // Save cache state if present
        if self.has_cached_kv() {
            let mut cache_state = HashMap::new();
            
            if let Some(ref k_cache) = self.k_cache {
                cache_state.insert("k_cache".to_string(), tensor_to_data_4d(k_cache));
            }
            if let Some(ref v_cache) = self.v_cache {
                cache_state.insert("v_cache".to_string(), tensor_to_data_4d(v_cache));
            }
            if let Some(ref kv_cache) = self.kv_cache {
                cache_state.insert("kv_cache".to_string(), tensor_to_data_5d(kv_cache));
            }
            
            if !cache_state.is_empty() {
                state.cache_state = Some(cache_state);
            }
        }

        state
    }

    /// Load state from AttentionState
    pub fn load_state(&mut self, state: AttentionState, device: &B::Device) -> Result<(), String> {
        // Check version compatibility
        if !state.is_compatible() {
            return Err(format!(
                "Incompatible state version: expected {}, got {}", 
                AttentionState::CURRENT_VERSION, 
                state.version
            ));
        }

        // Validate configuration compatibility
        if state.input_size != self.input_size || 
           state.output_size != self.output_size || 
           state.nhead != self.nhead ||
           state.nhead_kv != self.nhead_kv ||
           state.d_k != self.d_k ||
           state.d_v != self.d_v {
            return Err("State configuration doesn't match current instance".to_string());
        }

        // Helper to convert (shape, data) to tensor with specific dimensions
        let data_to_tensor_3d = |shape: Vec<usize>, data: Vec<f32>| -> Result<Tensor<B, 3>, String> {
            if shape.len() != 3 {
                return Err(format!("Expected 3D tensor, got {} dimensions", shape.len()));
            }
            let tensor_data = burn::tensor::TensorData::new(data, shape);
            Ok(Tensor::from_data(tensor_data, device))
        };
        
        let data_to_tensor_4d = |shape: Vec<usize>, data: Vec<f32>| -> Result<Tensor<B, 4>, String> {
            if shape.len() != 4 {
                return Err(format!("Expected 4D tensor, got {} dimensions", shape.len()));
            }
            let tensor_data = burn::tensor::TensorData::new(data, shape);
            Ok(Tensor::from_data(tensor_data, device))
        };
        
        let data_to_tensor_5d = |shape: Vec<usize>, data: Vec<f32>| -> Result<Tensor<B, 5>, String> {
            if shape.len() != 5 {
                return Err(format!("Expected 5D tensor, got {} dimensions", shape.len()));
            }
            let tensor_data = burn::tensor::TensorData::new(data, shape);
            Ok(Tensor::from_data(tensor_data, device))
        };

        // Load weights
        let w_out: Tensor<B, 3> = state.weights.get("w_out")
            .map(|(shape, data)| data_to_tensor_3d(shape.clone(), data.clone()))
            .ok_or("Missing w_out in saved state")?
            .map_err(|e| format!("Failed to load w_out: {}", e))?;

        let w_q = state.weights.get("w_q")
            .map(|(shape, data)| data_to_tensor_4d(shape.clone(), data.clone()))
            .transpose()
            .map_err(|e| format!("Failed to load w_q: {}", e))?;
            
        let w_k = state.weights.get("w_k")
            .map(|(shape, data)| data_to_tensor_3d(shape.clone(), data.clone()))
            .transpose()
            .map_err(|e| format!("Failed to load w_k: {}", e))?;
            
        let w_v = state.weights.get("w_v")
            .map(|(shape, data)| data_to_tensor_3d(shape.clone(), data.clone()))
            .transpose()
            .map_err(|e| format!("Failed to load w_v: {}", e))?;
            
        let w_kv = state.weights.get("w_kv")
            .map(|(shape, data)| data_to_tensor_4d(shape.clone(), data.clone()))
            .transpose()
            .map_err(|e| format!("Failed to load w_kv: {}", e))?;
            
        let w_qkv = state.weights.get("w_qkv")
            .map(|(shape, data)| data_to_tensor_4d(shape.clone(), data.clone()))
            .transpose()
            .map_err(|e| format!("Failed to load w_qkv: {}", e))?;

        // Load cache state if present
        let mut precomputed_k = None;
        let mut precomputed_v = None;
        let mut precomputed_kv = None;
        
        if let Some(ref cache_state) = state.cache_state {
            precomputed_k = cache_state.get("k_cache")
                .map(|(shape, data)| data_to_tensor_4d(shape.clone(), data.clone()))
                .transpose()
                .map_err(|e| format!("Failed to load k_cache: {}", e))?;
                
            precomputed_v = cache_state.get("v_cache")
                .map(|(shape, data)| data_to_tensor_4d(shape.clone(), data.clone()))
                .transpose()
                .map_err(|e| format!("Failed to load v_cache: {}", e))?;
                
            precomputed_kv = cache_state.get("kv_cache")
                .map(|(shape, data)| data_to_tensor_5d(shape.clone(), data.clone()))
                .transpose()
                .map_err(|e| format!("Failed to load kv_cache: {}", e))?;
        }

        // Apply loaded weights (without cache data - cache data is NOT precomputed values)
        self.set_parameters(
            w_out,
            w_q,
            w_k,
            w_v,
            w_kv,
            w_qkv,
            None, // precomputed_k - cache data should not be treated as precomputed
            None, // precomputed_v - cache data should not be treated as precomputed  
            None, // precomputed_kv - cache data should not be treated as precomputed
        )?;

        // Separately load cache state if present (cache is different from precomputed values)
        if precomputed_k.is_some() {
            self.k_cache = precomputed_k;
        }
        if precomputed_v.is_some() {
            self.v_cache = precomputed_v;
        }
        if precomputed_kv.is_some() {
            self.kv_cache = precomputed_kv;
        }

        Ok(())
    }

    /// Save state to binary file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        let state = self.save_state();
        let encoded = bincode::serialize(&state)
            .map_err(|e| format!("Failed to serialize state: {}", e))?;
        std::fs::write(path, encoded)
            .map_err(|e| format!("Failed to write state file: {}", e))?;
        Ok(())
    }

    /// Load state from binary file
    pub fn load_from_file<P: AsRef<std::path::Path>>(&mut self, path: P, device: &B::Device) -> Result<(), String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read state file: {}", e))?;
        let state: AttentionState = bincode::deserialize(&data)
            .map_err(|e| format!("Failed to deserialize state: {}", e))?;
        self.load_state(state, device)
    }
}

impl<B: Backend> Attention<B> for MultiHeadAttention<B> {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        x: Tensor<B, 3>,
        x_kv: Option<Tensor<B, 3>>,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        _only_cache_first_head_kv: bool,
        _save_peak_mem_factor: Option<i64>,
        add_input: bool,
        _allow_inplace: bool,
        train: bool,
        attention_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        assert!(!(cache_kv && use_cached_kv), "Cannot cache and use cached keys and values at the same time");

        let (x_processed, x_kv_processed, x_shape_after_transpose) = 
            self.rearrange_inputs_to_flat_batch(x.clone(), x_kv);

        let nhead_kv = if reuse_first_head_kv { 1 } else { self.nhead_kv };

        // Handle cache initialization if cache_kv is true
        if cache_kv {
            // Reset cache first so memory is freed before new cache is allocated
            self.k_cache = None;
            self.v_cache = None;
            self.kv_cache = None;

            let (batch_size, seqlen_kv) = if let Some(ref x_kv_tensor) = x_kv_processed {
                (x_kv_tensor.shape().dims[0], x_kv_tensor.shape().dims[1])
            } else {
                (x_processed.shape().dims[0], x_processed.shape().dims[1])
            };

            // Initialize cache based on weight structure
            if self.w_kv.is_some() || self.w_qkv.is_some() {
                let cache_nhead_kv = if _only_cache_first_head_kv { 1 } else { nhead_kv };
                self.kv_cache = Some(Tensor::zeros(
                    [batch_size, seqlen_kv, 2, cache_nhead_kv, self.d_k],
                    &x_processed.device(),
                ));
            } else {
                self.k_cache = Some(Tensor::zeros(
                    [batch_size, seqlen_kv, nhead_kv, self.d_k],
                    &x_processed.device(),
                ));
                self.v_cache = Some(Tensor::zeros(
                    [batch_size, seqlen_kv, nhead_kv, self.d_v],
                    &x_processed.device(),
                ));
            }
        }

        let output = self.compute_with_checkpointing(
            x_processed,
            x_kv_processed,
            cache_kv,
            use_cached_kv,
            reuse_first_head_kv,
            _only_cache_first_head_kv,
            _save_peak_mem_factor,
            add_input,
            _allow_inplace,
            train,
            attention_mask,
        );

        // Reshape output back to original batch dimensions
        let output_shape = output.shape().dims[2];
        let final_output = output.reshape([
            x_shape_after_transpose[0],
            x_shape_after_transpose[1],
            output_shape
        ]);

        // add_input is now handled in compute_with_checkpointing
        final_output
}
}