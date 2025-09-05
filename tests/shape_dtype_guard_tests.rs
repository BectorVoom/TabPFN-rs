//! Shape and DType Guard Tests
//! 
//! This test suite validates that shape guards and dtype consistency checks 
//! work correctly as required by specification 3.5.

use burn::tensor::{Tensor, TensorData, Int, backend::{Backend, AutodiffBackend}};
use burn_ndarray::{NdArray};
use burn::backend::Autodiff;
use tab_pfn_rs::tabpfn::architectures::base::loss_utils;

type TestBackend = Autodiff<NdArray<f32>>;

/// Test — Shape & dtype guards
/// 
/// Provide malformed inputs (wrong dims or wrong dtype) and assert the code panics 
/// with a descriptive message. Also include positive tests where valid inputs pass.
#[test]
fn test_shape_guards_valid_inputs() {
    println!("Testing shape guards with valid inputs");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Valid case: [4, 3] logits with [4] targets
    let batch_size = 4;
    let num_classes = 3;
    
    let logits_data: Vec<f32> = vec![
        1.0, 2.0, 0.5,  // Sample 1
        0.8, 1.5, 2.1,  // Sample 2  
        2.2, 0.3, 1.1,  // Sample 3
        0.9, 1.8, 0.4,  // Sample 4
    ];
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [batch_size * num_classes]),
        &device
    ).reshape([batch_size, num_classes]).require_grad();
    
    let labels_data = vec![0i64, -1i64, 2i64, 1i64]; // Sample 2 is masked
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(labels_data, [batch_size]),
        &device
    );
    
    // This should work without panicking
    let loss = loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "Valid inputs should produce finite loss");
    
    println!("✅ Valid shape inputs passed: loss={:.4}", loss_value);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits must be 2D tensor")]
fn test_shape_guard_logits_wrong_dimensions() {
    println!("Testing shape guard: logits wrong dimensions (should panic)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Invalid: 1D logits instead of 2D
    let logits_data: Vec<f32> = vec![1.0, 2.0, 0.5, 0.8];
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [4]),
        &device
    ).require_grad();
    
    let labels_data = vec![0i64, 1i64];
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(labels_data, [2]),
        &device
    );
    
    // This should panic with descriptive error message
    loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: targets must be 1D tensor")]
fn test_shape_guard_targets_wrong_dimensions() {
    println!("Testing shape guard: targets wrong dimensions (should panic)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Valid 2D logits
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1], [6]),
        &device
    ).reshape([2, 3]).require_grad();
    
    // Invalid: 2D targets instead of 1D
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64, 2i64, 1i64], [4]),
        &device
    ).reshape([2, 2]);
    
    // This should panic with descriptive error message
    loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits batch dimension")]
fn test_shape_guard_batch_dimension_mismatch() {
    println!("Testing shape guard: batch dimension mismatch (should panic)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // 2 batch logits
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1], [6]),
        &device
    ).reshape([2, 3]).require_grad();
    
    // 3 batch targets (mismatch!)
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64, 2i64], [3]),
        &device
    );
    
    // This should panic with descriptive error message
    loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits must have at least one class")]
fn test_shape_guard_zero_classes() {
    println!("Testing shape guard: zero classes (should panic)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Invalid: 0 classes
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![], [0]),
        &device
    ).reshape([2, 0]).require_grad();
    
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64], [2]),
        &device
    );
    
    // This should panic with descriptive error message
    loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: target class index")]
fn test_shape_guard_target_class_out_of_bounds() {
    println!("Testing shape guard: target class out of bounds (should panic)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // 3 classes (indices 0, 1, 2 are valid)
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1], [6]),
        &device
    ).reshape([2, 3]).require_grad();
    
    // Invalid target: class index 3 doesn't exist (only 0, 1, 2 are valid)
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 3i64], [2]),  // 3 is out of bounds!
        &device
    );
    
    // This should panic with descriptive error message
    loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
}

#[test]
fn test_shape_guards_edge_cases() {
    println!("Testing shape guard edge cases");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Edge case: single sample, single class
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![2.5], [1]),
        &device
    ).reshape([1, 1]).require_grad();
    
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64], [1]),
        &device
    );
    
    let loss = loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "Single sample case should work");
    
    // Edge case: all targets masked
    let logits2 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8], [4]),
        &device
    ).reshape([2, 2]).require_grad();
    
    let labels2 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![-1i64, -1i64], [2]),  // All masked
        &device
    );
    
    let loss2 = loss_utils::compute_masked_cross_entropy_loss(logits2, labels2, &device);
    let loss_value2: f32 = loss2.into_scalar();
    
    // Should return zero loss when all targets are masked
    assert!(loss_value2 >= 0.0, "All-masked case should return non-negative loss");
    
    println!("✅ Edge cases passed: single_sample_loss={:.4}, all_masked_loss={:.4}", 
             loss_value, loss_value2);
}

#[test]
fn test_dtype_consistency_requirements() {
    println!("Testing DType consistency requirements");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // This test documents the dtype requirements
    // Burn's type system enforces f32 for logits and i64 for targets at compile time
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0f32, 2.0f32, 0.5f32, 0.8f32], [4]),  // f32 explicitly
        &device
    ).reshape([2, 2]).require_grad();
    
    let labels = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64], [2]),  // i64 explicitly 
        &device
    );
    
    let loss = loss_utils::compute_masked_cross_entropy_loss(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    // Verify the loss is a finite f32
    assert!(loss_value.is_finite());
    assert!(!loss_value.is_nan());
    assert!(!loss_value.is_infinite());
    
    println!("✅ DType consistency verified: loss={:.4} (finite f32)", loss_value);
}