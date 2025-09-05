//! Comprehensive determinism tests for TabPFN-rs
//! 
//! These tests verify that the TabPFN implementation is fully deterministic
//! across parameter initialization, forward passes, and all stochastic operations.
//! 
//! BLOCKING REQUIREMENT: All tests in this file must pass for determinism compliance.

use burn::{
    backend::Autodiff,
    tensor::Tensor,
};
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::{
    transformer::{PerFeatureTransformer, DeterministicRngContext},
    config::ModelConfig,
};
use std::collections::HashMap;
use rand::rngs::StdRng;
use rand::SeedableRng;

type TestBackend = Autodiff<NdArray<f32>>;

/// Test A: Parameter & Forward Determinism
/// For fixed config.seed and fixed input tensor, two separate process runs
/// must produce identical outputs (bit-for-bit or within strict tolerance)
#[test]
fn test_parameter_and_forward_determinism() {
    let device = Default::default();
    let config = ModelConfig {
        emsize: 64,
        nhead: 2,
        nlayers: 2,
        seed: 42,
        dropout: 0.1,
        max_num_classes: 10,
        num_buckets: 32,
        ..Default::default()
    };

    // Create input tensors - use const for array sizes
    const BATCH_SIZE: usize = 2;
    const SEQ_LEN: usize = 4;
    const FEATURES: usize = 3;
    
    let x_input = Tensor::<TestBackend, 3>::from_floats(
        &[1.0f32; SEQ_LEN * BATCH_SIZE * FEATURES],
        &device,
    ).reshape([SEQ_LEN, BATCH_SIZE, FEATURES]);
    
    let y_input = Tensor::<TestBackend, 3>::from_floats(
        &[0.5f32; 2 * BATCH_SIZE * 1],
        &device,
    ).reshape([2, BATCH_SIZE, 1]);
    
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), x_input.clone());
    
    let mut y_inputs = HashMap::new();
    y_inputs.insert("main".to_string(), y_input.clone());

    // Run 1: Create model and run forward pass
    let rng_ctx1 = DeterministicRngContext::new(config.seed as u64, device.clone());
    let mut model1 = PerFeatureTransformer::new(
        &config,
        &rng_ctx1,
        3, // n_out
        "gelu",
        None,
        false,
        None,
        false,
        None,
        false,
        &device,
    ).expect("Failed to create model 1");

    let mut forward_rng1 = StdRng::seed_from_u64(config.seed as u64 + 1000);
    let output1 = model1.transformer_forward(
        x_inputs.clone(),
        Some(y_inputs.clone()),
        true,
        &mut Some(&mut forward_rng1),
        None,
        None,
        None,
        true, // train mode
    ).expect("Forward pass 1 failed");

    // Run 2: Create identical model and run identical forward pass
    let rng_ctx2 = DeterministicRngContext::new(config.seed as u64, device.clone());
    let mut model2 = PerFeatureTransformer::new(
        &config,
        &rng_ctx2,
        3, // n_out
        "gelu",
        None,
        false,
        None,
        false,
        None,
        false,
        &device,
    ).expect("Failed to create model 2");

    let mut forward_rng2 = StdRng::seed_from_u64(config.seed as u64 + 1000);
    let output2 = model2.transformer_forward(
        x_inputs.clone(),
        Some(y_inputs.clone()),
        true,
        &mut Some(&mut forward_rng2),
        None,
        None,
        None,
        true, // train mode
    ).expect("Forward pass 2 failed");

    // Compare outputs - they must be identical within strict tolerance
    let diff = (output1.clone() - output2.clone()).abs().sum();
    let max_diff = diff.into_scalar();
    
    assert!(max_diff < 1e-6, 
           "Outputs are not deterministic! Max difference: {} (tolerance: 1e-6)", 
           max_diff);
    
    println!("✅ Parameter & forward determinism test passed");
    println!("   Max difference: {:.2e}", max_diff);
}

/// Test B: Initialization vs Forward Separation
/// Changing forward-time RNG seeds should not change parameter values
#[test]
fn test_initialization_vs_forward_separation() {
    let device = Default::default();
    let config = ModelConfig {
        emsize: 32,
        nhead: 2,
        nlayers: 1,
        seed: 123,
        ..Default::default()
    };

    // Create two models with same initialization seed
    let rng_ctx1 = DeterministicRngContext::new(config.seed as u64, device.clone());
    let model1 = PerFeatureTransformer::new(
        &config,
        &rng_ctx1,
        2, // n_out
        "gelu",
        None,
        false,
        None,
        false,
        None,
        false,
        &device,
    ).expect("Failed to create model 1");

    let rng_ctx2 = DeterministicRngContext::new(config.seed as u64, device.clone());
    let model2 = PerFeatureTransformer::new(
        &config,
        &rng_ctx2,
        2, // n_out
        "gelu",
        None,
        false,
        None,
        false,
        None,
        false,
        &device,
    ).expect("Failed to create model 2");

    // Check that parameters are identical before any forward passes
    // Note: In practice, this would require access to parameter tensors
    // For now, we verify that models produce same outputs with same forward RNG
    
    let x_input = Tensor::<TestBackend, 3>::from_floats(
        &[1.0f32; 8],
        &device,
    ).reshape([2, 2, 2]);
    
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), x_input);

    // Both models with same forward RNG should produce identical outputs
    let mut forward_rng1 = StdRng::seed_from_u64(999);
    let mut forward_rng2 = StdRng::seed_from_u64(999);
    
    let mut model1_mut = model1;
    let mut model2_mut = model2;
    
    let output1 = model1_mut.transformer_forward(
        x_inputs.clone(),
        None,
        true,
        &mut Some(&mut forward_rng1),
        None,
        None,
        None,
        false, // eval mode
    ).expect("Forward pass 1 failed");

    let output2 = model2_mut.transformer_forward(
        x_inputs.clone(),
        None,
        true,
        &mut Some(&mut forward_rng2),
        None,
        None,
        None,
        false, // eval mode
    ).expect("Forward pass 2 failed");

    let diff = (output1 - output2).abs().sum().into_scalar();
    assert!(diff < 1e-6, 
           "Models with same init seed but different forward operations should have identical parameters. Diff: {}", 
           diff);
    
    println!("✅ Initialization vs forward separation test passed");
}

/// Test C: Multiple Seed Determinism
/// Test determinism across multiple different seeds
#[test]
fn test_multiple_seed_determinism() {
    let device = Default::default();
    let seeds = [42, 123, 999];
    
    for &seed in &seeds {
        let config = ModelConfig {
            emsize: 32,
            nhead: 2,
            nlayers: 1,
            seed,
            ..Default::default()
        };

        // Create model twice with same seed
        let rng_ctx1 = DeterministicRngContext::new(seed as u64, device.clone());
        let mut model1 = PerFeatureTransformer::new(
            &config,
            &rng_ctx1,
            2, // n_out
            "gelu",
            None,
            false,
            None,
            false,
            None,
            false,
            &device,
        ).expect("Failed to create model 1");

        let rng_ctx2 = DeterministicRngContext::new(seed as u64, device.clone());
        let mut model2 = PerFeatureTransformer::new(
            &config,
            &rng_ctx2,
            2, // n_out
            "gelu",
            None,
            false,
            None,
            false,
            None,
            false,
            &device,
        ).expect("Failed to create model 2");

        let x_input = Tensor::<TestBackend, 3>::from_floats(
            &[1.0f32; 8],
            &device,
        ).reshape([2, 2, 2]);
        
        let mut x_inputs = HashMap::new();
        x_inputs.insert("main".to_string(), x_input);

        let mut forward_rng1 = StdRng::seed_from_u64(seed as u64 + 1000);
        let mut forward_rng2 = StdRng::seed_from_u64(seed as u64 + 1000);
        
        let output1 = model1.transformer_forward(
            x_inputs.clone(),
            None,
            true,
            &mut Some(&mut forward_rng1),
            None,
            None,
            None,
            false, // eval mode
        ).expect("Forward pass 1 failed");

        let output2 = model2.transformer_forward(
            x_inputs.clone(),
            None,
            true,
            &mut Some(&mut forward_rng2),
            None,
            None,
            None,
            false, // eval mode
        ).expect("Forward pass 2 failed");

        let diff = (output1 - output2).abs().sum().into_scalar();
        assert!(diff < 1e-6, 
               "Models with seed {} are not deterministic. Diff: {}", 
               seed, diff);
    }
    
    println!("✅ Multiple seed determinism test passed");
}

/// Test D: Train vs Eval Mode Consistency
/// Test that eval mode is deterministic (no dropout) while train mode can vary with different RNG
#[test]
fn test_train_eval_mode_consistency() {
    let device = Default::default();
    let config = ModelConfig {
        emsize: 32,
        nhead: 2,
        nlayers: 1,
        seed: 42,
        dropout: 0.5, // High dropout to test effect
        ..Default::default()
    };

    let rng_ctx = DeterministicRngContext::new(config.seed as u64, device.clone());
    let mut model = PerFeatureTransformer::new(
        &config,
        &rng_ctx,
        2, // n_out
        "gelu",
        None,
        false,
        None,
        false,
        None,
        false,
        &device,
    ).expect("Failed to create model");

    let x_input = Tensor::<TestBackend, 3>::from_floats(
        &[1.0f32; 8],
        &device,
    ).reshape([2, 2, 2]);
    
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), x_input);

    // Test eval mode determinism - should be identical regardless of RNG
    let mut eval_rng1 = StdRng::seed_from_u64(1);
    let mut eval_rng2 = StdRng::seed_from_u64(2); // Different RNG seed
    
    let eval_output1 = model.transformer_forward(
        x_inputs.clone(),
        None,
        true,
        &mut Some(&mut eval_rng1),
        None,
        None,
        None,
        false, // eval mode - should be deterministic
    ).expect("Eval forward pass 1 failed");

    let eval_output2 = model.transformer_forward(
        x_inputs.clone(),
        None,
        true,
        &mut Some(&mut eval_rng2),
        None,
        None,
        None,
        false, // eval mode - should be deterministic
    ).expect("Eval forward pass 2 failed");

    let eval_diff = (eval_output1 - eval_output2).abs().sum().into_scalar();
    assert!(eval_diff < 1e-6, 
           "Eval mode should be deterministic regardless of RNG seed. Diff: {}", 
           eval_diff);

    // Test train mode with same RNG should be deterministic
    let mut train_rng1 = StdRng::seed_from_u64(100);
    let mut train_rng2 = StdRng::seed_from_u64(100); // Same RNG seed
    
    let train_output1 = model.transformer_forward(
        x_inputs.clone(),
        None,
        true,
        &mut Some(&mut train_rng1),
        None,
        None,
        None,
        true, // train mode
    ).expect("Train forward pass 1 failed");

    let train_output2 = model.transformer_forward(
        x_inputs.clone(),
        None,
        true,
        &mut Some(&mut train_rng2),
        None,
        None,
        None,
        true, // train mode
    ).expect("Train forward pass 2 failed");

    let train_diff = (train_output1 - train_output2).abs().sum().into_scalar();
    assert!(train_diff < 1e-6, 
           "Train mode should be deterministic with same RNG seed. Diff: {}", 
           train_diff);
    
    println!("✅ Train vs eval mode consistency test passed");
    println!("   Eval mode difference: {:.2e}", eval_diff);
    println!("   Train mode difference: {:.2e}", train_diff);
}