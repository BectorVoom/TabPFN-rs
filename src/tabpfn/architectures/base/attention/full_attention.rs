//! Implements standard quadratic attention.

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation;
use burn::nn::{Dropout, DropoutConfig};


use crate::tabpfn::architectures::base::config::ModelConfig;
use super::Attention;

/// Constant indicating if flash attention is available
#[allow(dead_code)]
const HAVE_FLASH_ATTN: bool = true;

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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_k: usize,
        d_v: usize,
        device: &B::Device,
        config: &ModelConfig,
        share_kv_across_n_heads: usize,
        dropout_p: Option<f64>,
        softmax_scale: Option<f64>,
        initialize_output_to_zero: bool,
        precomputed_k: Option<Tensor<B, 4>>,
        precomputed_v: Option<Tensor<B, 4>>,
        precomputed_kv: Option<Tensor<B, 5>>,
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
            Param::from_tensor(Tensor::zeros([nhead, d_v, output_size], device))
        } else {
            let w_out_tensor = Tensor::random([nhead, d_v, output_size], 
                burn::tensor::Distribution::Normal(0.0, 1.0), device);
            // Apply Xavier uniform initialization
            let fan_in = d_v;
            let fan_out = output_size;
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            Param::from_tensor(w_out_tensor * limit)
        };

        // Determine weight structure based on dimensions and precomputed values
        let has_precomputed_kv = precomputed_kv.is_some() || precomputed_k.is_some();
        
        // Initialize cache from precomputed values
        let k_cache = precomputed_k;
        let v_cache = precomputed_v;
        let kv_cache = precomputed_kv;
        
        // Initialize dropout module if dropout_p is provided
        let dropout = dropout_p.map(|p| DropoutConfig::new(p).init());
        
        let (w_q, w_k, w_v, w_kv, w_qkv) = if d_k == d_v && nhead == nhead_kv && !has_precomputed_kv {
            // Use combined w_qkv
            let w_qkv = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                [3, nhead, d_k, input_size],
                nhead,
                device,
                init_gain
            )));
            (None, None, None, None, w_qkv)
        } else {
            // Use separate weights
            let w_q = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                [1, nhead, d_k, input_size],
                nhead,
                device,
                init_gain
            )));
            
            if !has_precomputed_kv {
                if d_k == d_v {
                    // Use combined w_kv
                    let w_kv = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                        [2, nhead_kv, d_k, input_size],
                        nhead,
                        device,
                        init_gain
                    )));
                    (w_q, None, None, w_kv, None)
                } else {
                    // Use separate w_k and w_v
                    let w_k = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                        [nhead_kv, d_k, input_size],
                        nhead,
                        device,
                        init_gain
                    )));
                    let w_v = Some(Param::from_tensor(Self::newly_initialized_input_weight(
                        [nhead_kv, d_v, input_size],
                        nhead,
                        device,
                        init_gain
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
            init_gain,
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
        device: &B::Device,
        init_gain: f64,
    ) -> Tensor<B, D> {
        assert!(dims.len() >= 3 && dims.len() <= 4);
        let d = dims[dims.len() - 2];
        let input_size = dims[dims.len() - 1];
        let std = (2.0 / (nhead * d + input_size) as f64).sqrt() * init_gain;
        let a = (3.0_f64).sqrt() * std;
        
        Tensor::random(dims, burn::tensor::Distribution::Uniform(-a, a), device)
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
                        computed_k = computed_k.repeat(&[1, 1, orig_heads, 1]);
                        computed_v = computed_v.repeat(&[1, 1, orig_heads, 1]);
                    }
                    
                    k = Some(computed_k);
                    v = Some(computed_v);
                }
            }
        }

        // *** NEW: CRITICAL MISSING FEATURE - Cache Update Implementation ***
        if cache_kv {
            // Update k_cache and v_cache if they exist and we have computed k,v
            if let (Some(k_cache_tensor), Some(computed_k)) = (&self.k_cache, &k) {
                // Validate shape compatibility
                let cache_shape = k_cache_tensor.shape().dims;
                let computed_shape = computed_k.shape().dims;
                if cache_shape != computed_shape {
                    panic!("Cache shape mismatch: k_cache {:?} vs computed_k {:?}", cache_shape, computed_shape);
                }
                // Update cache with computed values (equivalent to Python's k_cache[:] = k)
                self.k_cache = Some(computed_k.clone());
            }
            
            if let (Some(v_cache_tensor), Some(computed_v)) = (&self.v_cache, &v) {
                // Validate shape compatibility  
                let cache_shape = v_cache_tensor.shape().dims;
                let computed_shape = computed_v.shape().dims;
                if cache_shape != computed_shape {
                    panic!("Cache shape mismatch: v_cache {:?} vs computed_v {:?}", cache_shape, computed_shape);
                }
                // Update cache with computed values (equivalent to Python's v_cache[:] = v)
                self.v_cache = Some(computed_v.clone());
            }
            
            // Handle kv_cache updates with special logic for first-head-only caching
            if let (Some(kv_cache_tensor), Some(computed_kv)) = (&self.kv_cache, &kv) {
                let cache_shape = kv_cache_tensor.shape().dims;
                let computed_shape = computed_kv.shape().dims;
                
                // Check if this is first-head-only caching (cache shape has nhead_kv=1)
                if cache_shape.len() >= 2 && cache_shape[cache_shape.len()-2] == 1 {
                    // Only cache first head: equivalent to Python's kv_cache[:] = kv[..., :1, :]
                    let first_head_kv = computed_kv.clone().slice([
                        0..computed_shape[0], 
                        0..computed_shape[1], 
                        0..computed_shape[2], 
                        0..1,  // Only first head
                        0..computed_shape[4]
                    ]);
                    let first_head_shape = first_head_kv.shape().dims;
                    if cache_shape != first_head_shape {
                        panic!("Cache shape mismatch: kv_cache {:?} vs first_head_kv {:?}", cache_shape, first_head_shape);
                    }
                    self.kv_cache = Some(first_head_kv);
                } else {
                    // Cache all heads: equivalent to Python's kv_cache[:] = kv
                    if cache_shape != computed_shape {
                        panic!("Cache shape mismatch: kv_cache {:?} vs computed_kv {:?}", cache_shape, computed_shape);
                    }
                    self.kv_cache = Some(computed_kv.clone());
                }
            }
        }

        (Some(q), k, v, kv, None)
    }

    fn einsum_q(&self, x: Tensor<B, 3>, w_q: Tensor<B, 3>) -> Tensor<B, 4> {
        // Equivalent to: torch.einsum("... s, h d s -> ... h d", x, w_q)
        let batch_size = x.shape().dims[0];
        let seq_len = x.shape().dims[1];
        
        // Reshape x to [batch*seq, input_size] and w_q to [nhead*d_k, input_size]
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_q_flat = w_q.reshape([self.nhead * self.d_k, self.input_size]);
        
        // Matrix multiply: [batch*seq, input_size] @ [input_size, nhead*d_k] -> [batch*seq, nhead*d_k]
        let result_flat = x_flat.matmul(w_q_flat.transpose());
        
        // Reshape back to [batch, seq, nhead, d_k]
        result_flat.reshape([batch_size, seq_len, self.nhead, self.d_k])
    }

    fn einsum_k(&self, x: Tensor<B, 3>, w_k: Tensor<B, 3>) -> Tensor<B, 4> {
        // Equivalent to: torch.einsum("... s, h d s -> ... h d", x, w_k)
        let batch_size = x.shape().dims[0];
        let seq_len = x.shape().dims[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_k_flat = w_k.reshape([self.nhead_kv * self.d_k, self.input_size]);
        
        let result_flat = x_flat.matmul(w_k_flat.transpose());
        result_flat.reshape([batch_size, seq_len, self.nhead_kv, self.d_k])
    }

    fn einsum_v(&self, x: Tensor<B, 3>, w_v: Tensor<B, 3>) -> Tensor<B, 4> {
        // Equivalent to: torch.einsum("... s, h d s -> ... h d", x, w_v)
        let batch_size = x.shape().dims[0];
        let seq_len = x.shape().dims[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_v_flat = w_v.reshape([self.nhead_kv * self.d_v, self.input_size]);
        
        let result_flat = x_flat.matmul(w_v_flat.transpose());
        result_flat.reshape([batch_size, seq_len, self.nhead_kv, self.d_v])
    }

    fn einsum_kv(&self, x: Tensor<B, 3>, w_kv: Tensor<B, 4>) -> Tensor<B, 5> {
        // Equivalent to: torch.einsum("... s, j h d s -> ... j h d", x, w_kv)
        let batch_size = x.shape().dims[0];
        let seq_len = x.shape().dims[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_kv_flat = w_kv.reshape([2 * self.nhead_kv * self.d_k, self.input_size]);
        
        let result_flat = x_flat.matmul(w_kv_flat.transpose());
        result_flat.reshape([batch_size, seq_len, 2, self.nhead_kv, self.d_k])
    }

    fn einsum_qkv(&self, x: Tensor<B, 3>, w_qkv: Tensor<B, 4>) -> Tensor<B, 5> {
        // Equivalent to: torch.einsum("... s, j h d s -> ... j h d", x, w_qkv)
        let batch_size = x.shape().dims[0];
        let seq_len = x.shape().dims[1];
        
        let x_flat = x.reshape([batch_size * seq_len, self.input_size]);
        let w_qkv_flat = w_qkv.reshape([3 * self.nhead * self.d_k, self.input_size]);
        
        let result_flat = x_flat.matmul(w_qkv_flat.transpose());
        result_flat.reshape([batch_size, seq_len, 3, self.nhead, self.d_k])
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
    ) -> Tensor<B, 4> {
        assert!(k.is_none() == v.is_none());
        assert!([qkv.is_none(), kv.is_none(), k.is_none() && v.is_none()].iter().filter(|&&x| x).count() == 2);
        assert!(qkv.is_none() != q.is_none());

        let (q, k, v) = if let Some(qkv_tensor) = qkv {
            let qkv_sliced = qkv_tensor.clone();
            let q = qkv_sliced.clone().slice([0..qkv_sliced.shape().dims[0], 0..qkv_sliced.shape().dims[1], 0..1, 0..qkv_sliced.shape().dims[3], 0..qkv_sliced.shape().dims[4]]).squeeze::<4>(2);
            let k = qkv_sliced.clone().slice([0..qkv_sliced.shape().dims[0], 0..qkv_sliced.shape().dims[1], 1..2, 0..qkv_sliced.shape().dims[3], 0..qkv_sliced.shape().dims[4]]).squeeze::<4>(2);
            let v = qkv_sliced.clone().slice([0..qkv_sliced.shape().dims[0], 0..qkv_sliced.shape().dims[1], 2..3, 0..qkv_sliced.shape().dims[3], 0..qkv_sliced.shape().dims[4]]).squeeze::<4>(2);
            (q, k, v)
        } else if let Some(kv_tensor) = kv {
            let q = q.unwrap();
            let k = kv_tensor.clone().slice([0..kv_tensor.shape().dims[0], 0..kv_tensor.shape().dims[1], 0..1, 0..kv_tensor.shape().dims[3], 0..kv_tensor.shape().dims[4]]).squeeze::<4>(2);
            let v = kv_tensor.clone().slice([0..kv_tensor.shape().dims[0], 0..kv_tensor.shape().dims[1], 1..2, 0..kv_tensor.shape().dims[3], 0..kv_tensor.shape().dims[4]]).squeeze::<4>(2);
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
        let logits_scaled = logits * scale;

        let mut ps = activation::softmax(logits_scaled, 3); // softmax over seq_kv dimension
        
        // Apply dropout to attention probabilities if specified
        if let Some(dropout_module) = dropout {
            ps = dropout_module.forward(ps);
        }

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
    ) -> Tensor<B, 3> {
        let (q, k, v, kv, qkv) = self.compute_qkv(
            x,
            x_kv,
            cache_kv,
            use_cached_kv,
            reuse_first_head_kv,
        );

        let attention_head_outputs = Self::compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            self.dropout.as_ref(),
            self.softmax_scale,
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
            
            // Compute attention for this chunk
            let chunk_output = self.compute(
                x_chunk.clone(),
                x_kv_chunk,
                false, // Don't cache KV for chunks to save memory
                use_cached_kv,
                reuse_first_head_kv,
                only_cache_first_head_kv,
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