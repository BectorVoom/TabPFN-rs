//! Base attention trait and implementations

use burn::prelude::*;

/// Base trait for attention layers
pub trait Attention<B: Backend> {
    /// Performs the attention layer computation
    /// 
    /// # Arguments
    /// * `attention_mask` - [batch, seq_q, seq_kv] mask where 1.0 = allowed, 0.0 = masked (prevents label leakage)
    #[allow(clippy::too_many_arguments)]
    fn forward(
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
    ) -> Tensor<B, 3>;
}

pub mod full_attention;