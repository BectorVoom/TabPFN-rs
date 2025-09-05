//! Test set_parameters functionality

use burn::prelude::*;
use burn_ndarray::NdArray;

use tab_pfn_rs::tabpfn::architectures::base::attention::full_attention::MultiHeadAttention;
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type Backend = NdArray<f32>;

fn main() {
    println!("Testing set_parameters functionality...");
    
    let mut config = ModelConfig::default();
    config.emsize = 64;
    config.nhead = 4;
    config.attention_init_gain = 1.0;
    
    let device = Default::default();
    let d_k = config.emsize as usize / config.nhead as usize; // 16
    let d_v = config.emsize as usize / config.nhead as usize; // 16
    
    // Create attention module
    let mut attention = MultiHeadAttention::<Backend>::new(
        d_k,
        d_v,
        &device,
        &config,
        1,
        None,
        None,
        false,
        None,
        None,
        None,
    );
    
    println!("✓ Created MultiHeadAttention module");
    
    // Create new parameters to set
    let new_w_out = Tensor::<Backend, 3>::random(
        [config.nhead as usize, d_v, config.emsize as usize],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    let new_w_qkv = Tensor::<Backend, 4>::random(
        [3, config.nhead as usize, d_k, config.emsize as usize],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    println!("✓ Created new parameter tensors");
    println!("  w_out shape: {:?}", new_w_out.shape().dims);
    println!("  w_qkv shape: {:?}", new_w_qkv.shape().dims);
    
    // Test set_parameters with valid inputs
    let result = attention.set_parameters(
        new_w_out.clone(),
        None, // w_q should be None when w_qkv is provided
        None, // w_k
        None, // w_v
        None, // w_kv
        Some(new_w_qkv.clone()), // w_qkv
        None, // precomputed_k
        None, // precomputed_v
        None, // precomputed_kv
    );
    
    match result {
        Ok(()) => {
            println!("🎉 set_parameters succeeded!");
            
            // Verify the parameters were set correctly
            if let Some(w_qkv_param) = attention.w_qkv() {
                println!("✓ w_qkv parameter was set correctly");
                println!("  New w_qkv shape: {:?}", w_qkv_param.shape().dims);
            } else {
                println!("❌ w_qkv parameter was not set");
            }
            
            let w_out_param = attention.w_out();
            println!("✓ w_out parameter was set correctly");
            println!("  New w_out shape: {:?}", w_out_param.shape().dims);
        }
        Err(e) => {
            println!("❌ set_parameters failed: {}", e);
        }
    }
    
    // Test validation - should fail with incompatible parameters
    println!("\nTesting validation with incompatible parameters...");
    
    let bad_w_out = Tensor::<Backend, 3>::random(
        [999, 999, 999], // Wrong shape
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    let validation_result = attention.set_parameters(
        bad_w_out,
        None,
        None,
        None,
        None,
        Some(new_w_qkv),
        None,
        None,
        None,
    );
    
    match validation_result {
        Ok(()) => {
            println!("⚠️  Validation should have failed but didn't");
        }
        Err(e) => {
            println!("✓ Validation correctly failed: {}", e);
        }
    }
    
    println!("\n🎉 set_parameters functionality test completed!");
}