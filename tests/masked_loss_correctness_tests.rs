//! Masked Loss Correctness Tests
//! 
//! These tests verify that the masked cross-entropy loss implementation correctly
//! handles ignore_index=-1 values and produces numerically accurate results.

use burn::tensor::{Tensor, TensorData, Int, backend::{Backend, AutodiffBackend}};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::nn::loss::CrossEntropyLoss;
use tab_pfn_rs::tabpfn::architectures::base::loss_utils::{compute_masked_cross_entropy_loss, compute_masked_cross_entropy_loss_ignore_index};

type TestBackend = Autodiff<NdArray<f32>>;

/// Test — Masked loss excludes ignore_index=-1 correctly
/// 
/// Create logits and targets where some targets are -1 (ignore_index).
/// Verify that the computed loss matches manual computation on non-masked targets only.
#[test]
fn test_masked_loss_excludes_ignore_index() {
    println!("Running Test: Masked loss excludes ignore_index=-1 correctly");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test data: 4 samples, 3 classes
    // Sample 0: target=1, Sample 1: target=-1 (ignore), Sample 2: target=0, Sample 3: target=2
    let logits_data = vec![
        // Sample 0: [1.0, 2.0, 0.5] - target should be 1
        1.0, 2.0, 0.5,
        // Sample 1: [0.8, 1.5, 2.1] - target is -1 (ignore)
        0.8, 1.5, 2.1,
        // Sample 2: [2.5, 0.3, 0.7] - target should be 0  
        2.5, 0.3, 0.7,
        // Sample 3: [0.1, 0.4, 1.8] - target should be 2
        0.1, 0.4, 1.8,
    ];
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data, [4, 3]),
        &device
    );
    
    let targets_data = vec![1i64, -1i64, 0i64, 2i64];
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [4]),
        &device
    );
    
    // Compute masked loss
    let masked_loss = compute_masked_cross_entropy_loss_ignore_index(logits.clone(), targets.clone(), &device);
    let masked_loss_value: f32 = masked_loss.into_scalar();
    
    // Manually compute expected loss on non-masked samples (indices 0, 2, 3)
    let valid_logits_data = vec![
        1.0, 2.0, 0.5,  // Sample 0: target=1
        2.5, 0.3, 0.7,  // Sample 2: target=0
        0.1, 0.4, 1.8,  // Sample 3: target=2
    ];
    let valid_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(valid_logits_data, [3, 3]),
        &device
    );
    
    let valid_targets_data = vec![1i64, 0i64, 2i64];
    let valid_targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(valid_targets_data, [3]),
        &device
    );
    
    // Compute reference loss using standard CrossEntropyLoss
    let reference_loss_fn = CrossEntropyLoss::new(None, &device);
    let reference_loss = reference_loss_fn.forward(valid_logits, valid_targets);
    let reference_loss_value: f32 = reference_loss.into_scalar();
    
    println!("   Masked loss value: {:.6}", masked_loss_value);
    println!("   Reference loss value: {:.6}", reference_loss_value);
    
    // Verify that masked loss matches reference loss within tolerance
    let tolerance = 1e-5;
    let difference = (masked_loss_value - reference_loss_value).abs();
    assert!(difference < tolerance, 
            "Masked loss should match reference loss on valid samples. \
            Masked: {:.6}, Reference: {:.6}, Difference: {:.6}",
            masked_loss_value, reference_loss_value, difference);
    
    // Verify both losses are finite and positive
    assert!(masked_loss_value.is_finite(), "Masked loss should be finite");
    assert!(reference_loss_value.is_finite(), "Reference loss should be finite");
    assert!(masked_loss_value > 0.0, "Loss should be positive");
    
    println!("✅ Test PASSED: Masked loss correctly excludes ignore_index=-1");
    println!("   Loss difference: {:.8} (< {:.8})", difference, tolerance);
}

/// Test — All targets masked should panic
/// 
/// When all targets are -1 (ignore_index), there are no valid examples 
/// to compute loss on, so the function should panic with a clear error.
#[test]
#[should_panic(expected = "Masked loss: no valid positions in mask")]
fn test_all_targets_masked_should_panic() {
    println!("Running Test: All targets masked should panic");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test data where all targets are masked
    let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1];
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data, [2, 3]),
        &device
    );
    
    let targets_data = vec![-1i64, -1i64]; // All targets masked
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [2]),
        &device
    );
    
    // Should panic with clear error message
    compute_masked_cross_entropy_loss_ignore_index(logits, targets, &device);
}

/// Test — No targets masked behaves like standard cross-entropy
/// 
/// When no targets are masked (no -1 values), the masked loss should
/// behave identically to standard CrossEntropyLoss.
#[test]
fn test_no_targets_masked_matches_standard_loss() {
    println!("Running Test: No targets masked matches standard loss");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test data with no masked targets
    let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1, 2.5, 0.3, 0.7];
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data, [3, 3]),
        &device
    );
    
    let targets_data = vec![1i64, 2i64, 0i64]; // No -1 values
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [3]),
        &device
    );
    
    // Compute masked loss
    let masked_loss = compute_masked_cross_entropy_loss_ignore_index(logits.clone(), targets.clone(), &device);
    let masked_loss_value: f32 = masked_loss.into_scalar();
    
    // Compute standard loss
    let standard_loss_fn = CrossEntropyLoss::new(None, &device);
    let standard_loss = standard_loss_fn.forward(logits, targets);
    let standard_loss_value: f32 = standard_loss.into_scalar();
    
    println!("   Masked loss (no masking): {:.6}", masked_loss_value);
    println!("   Standard loss: {:.6}", standard_loss_value);
    
    // They should be identical (within floating point precision)
    let tolerance = 1e-6;
    let difference = (masked_loss_value - standard_loss_value).abs();
    assert!(difference < tolerance,
            "Masked loss without masking should match standard CrossEntropyLoss. \
            Masked: {:.6}, Standard: {:.6}, Difference: {:.6}",
            masked_loss_value, standard_loss_value, difference);
    
    println!("✅ Test PASSED: No masking produces identical results to standard loss");
    println!("   Loss difference: {:.8} (< {:.8})", difference, tolerance);
}

/// Test — Mixed masking scenarios
/// 
/// Test various combinations of masked and unmasked targets to ensure
/// robustness across different masking patterns.
#[test]
fn test_mixed_masking_scenarios() {
    println!("Running Test: Mixed masking scenarios");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Scenario 1: First and last masked
    let logits1 = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1, 2.5, 0.3, 0.7, 0.1, 0.4, 1.8], [4, 3]),
        &device
    );
    let targets1 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![-1i64, 1i64, 0i64, -1i64], [4]),
        &device
    );
    
    let loss1 = compute_masked_cross_entropy_loss_ignore_index(logits1, targets1, &device);
    let loss1_value: f32 = loss1.into_scalar();
    
    // Scenario 2: Middle two masked
    let logits2 = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1, 2.5, 0.3, 0.7, 0.1, 0.4, 1.8], [4, 3]),
        &device
    );
    let targets2 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64, -1i64, -1i64, 2i64], [4]),
        &device
    );
    
    let loss2 = compute_masked_cross_entropy_loss_ignore_index(logits2, targets2, &device);
    let loss2_value: f32 = loss2.into_scalar();
    
    // Scenario 3: Alternating pattern
    let logits3 = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1, 2.5, 0.3, 0.7, 0.1, 0.4, 1.8, 1.2, 0.9, 1.1, 0.6, 0.2, 1.4], [6, 3]),
        &device
    );
    let targets3 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, -1i64, 1i64, -1i64, 2i64, -1i64], [6]),
        &device
    );
    
    let loss3 = compute_masked_cross_entropy_loss_ignore_index(logits3, targets3, &device);
    let loss3_value: f32 = loss3.into_scalar();
    
    println!("   Scenario 1 (first/last masked): {:.6}", loss1_value);
    println!("   Scenario 2 (middle masked): {:.6}", loss2_value);
    println!("   Scenario 3 (alternating): {:.6}", loss3_value);
    
    // All losses should be finite and positive
    assert!(loss1_value.is_finite() && loss1_value > 0.0, "Scenario 1 loss should be finite and positive");
    assert!(loss2_value.is_finite() && loss2_value > 0.0, "Scenario 2 loss should be finite and positive");
    assert!(loss3_value.is_finite() && loss3_value > 0.0, "Scenario 3 loss should be finite and positive");
    
    println!("✅ Test PASSED: All mixed masking scenarios produce valid losses");
}

/// Test — Gradient flow with masked targets
/// 
/// Verify that gradients can flow through the masked loss computation
/// and that the computational graph is preserved correctly.
#[test]
fn test_gradient_flow_with_masking() {
    println!("Running Test: Gradient flow with masked targets");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits that require gradients
    let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1];
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data, [2, 3]),
        &device
    ).require_grad();
    
    let targets_data = vec![1i64, -1i64]; // One valid, one masked
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [2]),
        &device
    );
    
    // Compute masked loss and backward pass
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits.clone(), targets, &device);
    let loss_value: f32 = loss.clone().into_scalar();
    
    // Verify we can compute gradients
    let gradients = loss.backward();
    let logits_grad = logits.grad(&gradients);
    
    // Check that gradients exist and have correct shape
    assert!(logits_grad.is_some(), "Gradients should be computed for input logits");
    
    let grad_tensor = logits_grad.unwrap();
    assert_eq!(grad_tensor.shape().dims, [2, 3], "Gradient tensor should have same shape as input");
    
    // Convert gradients to data to verify they're not all zero
    let grad_data = grad_tensor.into_data();
    let grad_values: Vec<f32> = grad_data.iter::<f32>().collect();
    
    // At least some gradients should be non-zero (for the non-masked sample)
    let non_zero_grads = grad_values.iter().filter(|&&x| x.abs() > 1e-8).count();
    assert!(non_zero_grads > 0, "Some gradients should be non-zero for valid samples");
    
    println!("   Loss value: {:.6}", loss_value);
    println!("   Non-zero gradients: {}/{}", non_zero_grads, grad_values.len());
    
    assert!(loss_value.is_finite() && loss_value > 0.0, "Loss should be finite and positive");
    
    println!("✅ Test PASSED: Gradient flow works correctly with masked targets");
}

/// Test — Single unmasked sample
/// 
/// Test behavior when there's one valid (unmasked) sample.
#[test]
fn test_single_unmasked_sample() {
    println!("Running Test: Single unmasked sample");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Single unmasked sample
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5], [1, 3]),
        &device
    );
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64], [1]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, targets, &device);
    let loss_value: f32 = loss.into_scalar();
    
    println!("   Single unmasked sample loss: {:.6}", loss_value);
    
    // Should be positive and finite
    assert!(loss_value.is_finite() && loss_value > 0.0, "Single unmasked sample should produce positive loss");
    
    println!("✅ Test PASSED: Single unmasked sample handled correctly");
}

/// Test — Single masked sample should panic
/// 
/// Test behavior when there's only one sample and it's masked (-1).
/// This should panic since there are no valid examples to compute loss on.
#[test]
#[should_panic(expected = "Masked loss: no valid positions in mask")]
fn test_single_masked_sample_should_panic() {
    println!("Running Test: Single masked sample should panic");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Single masked sample
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5], [1, 3]),
        &device
    );
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![-1i64], [1]),
        &device
    );
    
    // Should panic with clear error message
    compute_masked_cross_entropy_loss_ignore_index(logits, targets, &device);
}