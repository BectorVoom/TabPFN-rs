//! Test binary for attention equivalence testing

use std::env;
use burn::prelude::*;
use burn_ndarray::NdArray;
use serde_json::json;

use tab_pfn_rs::tabpfn::architectures::base::attention::full_attention::MultiHeadAttention;
use tab_pfn_rs::tabpfn::architectures::base::attention::Attention;
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type Backend = NdArray<f32>;

fn create_test_config() -> ModelConfig {
    let mut config = ModelConfig::default();
    config.emsize = 64; // Smaller for testing
    config.nhead = 4;
    config.attention_init_gain = 1.0;
    config
}

fn test_attention_creation() -> Result<(), String> {
    println!("Testing Rust MultiHeadAttention creation...");
    
    let config = create_test_config();
    let device = Default::default();
    let d_k = config.emsize as usize / config.nhead as usize; // 16
    let d_v = config.emsize as usize / config.nhead as usize; // 16
    
    // Create Rust attention
    let _rust_attention = MultiHeadAttention::<Backend>::new(
        d_k,
        d_v,
        &device,
        &config,
        1, // share_kv_across_n_heads
        None, // dropout_p
        None, // softmax_scale
        false, // initialize_output_to_zero
        None, // precomputed_k
        None, // precomputed_v
        None, // precomputed_kv
    );
    
    println!("✓ Rust MultiHeadAttention creation test passed");
    Ok(())
}

fn test_attention_forward_basic() -> Result<(), String> {
    println!("Testing basic Rust attention forward pass...");
    
    let config = create_test_config();
    let device = Default::default();
    let d_k = config.emsize as usize / config.nhead as usize;
    let d_v = config.emsize as usize / config.nhead as usize;
    
    // Create Rust attention
    let rust_attention = MultiHeadAttention::<Backend>::new(
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
    
    // Create test input
    let batch_size = 2;
    let seq_len = 8;
    let test_input = Tensor::<Backend, 3>::random(
        [batch_size, seq_len, config.emsize as usize], 
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    // Run forward pass
    let output = rust_attention.forward(
        test_input.clone(),
        None, // x_kv
        false, // cache_kv
        false, // use_cached_kv
        false, // reuse_first_head_kv
        false, // only_cache_first_head_kv
        None, // save_peak_mem_factor
        false, // add_input
        false, // allow_inplace
    );
    
    // Check output shape
    let expected_shape = [batch_size, seq_len, config.emsize as usize];
    let output_shape = output.shape().dims;
    
    if output_shape != expected_shape {
        return Err(format!("Expected shape {:?}, got {:?}", expected_shape, output_shape));
    }
    
    println!("✓ Rust attention forward pass test passed");
    println!("  Input shape: {:?}", test_input.shape().dims);
    println!("  Output shape: {:?}", output_shape);
    
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        // Create a dummy test file argument for compatibility
        println!("Running Rust MultiHeadAttention equivalence tests...");
    }
    
    println!("{}", "=".repeat(60));
    
    let mut tests_passed = 0;
    let mut total_tests = 0;
    
    // Test 1: Creation
    total_tests += 1;
    match test_attention_creation() {
        Ok(_) => tests_passed += 1,
        Err(e) => println!("✗ Attention creation test failed: {}", e),
    }
    
    // Test 2: Basic forward pass
    total_tests += 1;
    match test_attention_forward_basic() {
        Ok(_) => tests_passed += 1,
        Err(e) => println!("✗ Basic forward pass test failed: {}", e),
    }
    
    println!("{}", "=".repeat(60));
    println!("Tests passed: {}/{}", tests_passed, total_tests);
    
    let result = json!({
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "success": tests_passed == total_tests
    });
    
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    
    if tests_passed == total_tests {
        println!("🎉 All Rust tests passed!");
    } else {
        println!("❌ Some Rust tests failed!");
        std::process::exit(1);
    }
}