//! Test to verify dropout functionality works correctly

use burn::prelude::*;
use burn_ndarray::NdArray;

use tab_pfn_rs::tabpfn::architectures::base::attention::full_attention::MultiHeadAttention;
use tab_pfn_rs::tabpfn::architectures::base::attention::Attention;
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type Backend = NdArray<f32>;

fn main() {
    println!("Testing Dropout functionality in MultiHeadAttention...");
    
    let mut config = ModelConfig::default();
    config.emsize = 64;
    config.nhead = 4;
    config.attention_init_gain = 1.0;
    
    let device = Default::default();
    let d_k = config.emsize as usize / config.nhead as usize;
    let d_v = config.emsize as usize / config.nhead as usize;
    
    // Create attention with 0.1 dropout probability
    let mut rust_attention = MultiHeadAttention::<Backend>::new(
        d_k,
        d_v,
        &device,
        &config,
        1,
        Some(0.1), // Enable dropout
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
    
    // Run forward pass twice - outputs should be different due to dropout randomness
    let output1 = rust_attention.forward(
        test_input.clone(),
        None,
        false,
        false,
        false,
        false,
        None,
        false,
        false,
    );
    
    let output2 = rust_attention.forward(
        test_input.clone(),
        None,
        false,
        false,
        false,
        false,
        None,
        false,
        false,
    );
    
    // Compare outputs - they should be different due to dropout
    let diff = (output1.clone() - output2.clone()).abs().sum();
    let diff_scalar = diff.into_scalar();
    
    println!("✓ Created MultiHeadAttention with dropout_p = 0.1");
    println!("✓ Forward pass 1 shape: {:?}", output1.shape().dims);
    println!("✓ Forward pass 2 shape: {:?}", output2.shape().dims);
    println!("✓ Absolute difference sum: {:.6}", diff_scalar);
    
    if diff_scalar > 1e-6 {
        println!("🎉 Dropout is working! Outputs are different due to random dropout.");
    } else {
        println!("⚠️  Warning: Outputs are very similar. Dropout might not be active.");
    }
    
    // Test with no dropout for comparison
    let mut rust_attention_no_dropout = MultiHeadAttention::<Backend>::new(
        d_k,
        d_v,
        &device,
        &config,
        1,
        None, // No dropout
        None,
        false,
        None,
        None,
        None,
    );
    
    let output3 = rust_attention_no_dropout.forward(
        test_input.clone(),
        None,
        false,
        false,
        false,
        false,
        None,
        false,
        false,
    );
    
    let output4 = rust_attention_no_dropout.forward(
        test_input.clone(),
        None,
        false,
        false,
        false,
        false,
        None,
        false,
        false,
    );
    
    let no_dropout_diff = (output3.clone() - output4.clone()).abs().sum();
    let no_dropout_diff_scalar = no_dropout_diff.into_scalar();
    
    println!("✓ No dropout - difference sum: {:.10}", no_dropout_diff_scalar);
    
    if no_dropout_diff_scalar < 1e-9 {
        println!("🎉 Without dropout, outputs are identical as expected!");
    } else {
        println!("⚠️  Warning: Even without dropout, outputs are different.");
    }
    
    println!("🎉 Dropout functionality test completed!");
}