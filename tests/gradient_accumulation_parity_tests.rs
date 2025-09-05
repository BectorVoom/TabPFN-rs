//! Gradient Accumulation Parity Tests
//! 
//! These tests verify that the new loss accumulation approach produces equivalent
//! results to proper gradient accumulation, ensuring mathematical correctness.

use burn::tensor::{Tensor, TensorData, Int, backend::{Backend, AutodiffBackend}};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::nn::loss::CrossEntropyLoss;
use tab_pfn_rs::tabpfn::architectures::base::loss_utils::compute_masked_cross_entropy_loss_ignore_index;

type TestBackend = Autodiff<NdArray<f32>>;

/// Test — Loss accumulation equals gradient accumulation
/// 
/// This test verifies that:
/// 1. Loss accumulation + single backward pass 
/// 2. Equals: Sum of individual losses with individual backward passes
/// 
/// This ensures our architectural change maintains mathematical correctness.
#[test]
fn test_loss_accumulation_equals_gradient_accumulation() {
    println!("Running Test: Loss accumulation equals gradient accumulation");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create multiple batches of test data that we'll process
    // We'll simulate gradient accumulation across 3 mini-batches
    
    // Mini-batch 1: 2 samples, 3 classes
    let logits1_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1];
    let logits1 = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits1_data, [2, 3]),
        &device
    ).require_grad();
    let targets1 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64, 0i64], [2]),
        &device
    );
    
    // Mini-batch 2: 2 samples, 3 classes  
    let logits2_data = vec![2.5, 0.3, 0.7, 0.1, 0.4, 1.8];
    let logits2 = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits2_data, [2, 3]),
        &device
    ).require_grad();
    let targets2 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 2i64], [2]),
        &device
    );
    
    // Mini-batch 3: 2 samples, 3 classes (with one masked)
    let logits3_data = vec![1.2, 0.9, 1.1, 0.6, 0.2, 1.4];
    let logits3 = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits3_data, [2, 3]),
        &device
    ).require_grad();
    let targets3 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![2i64, -1i64], [2]), // Second target masked
        &device
    );
    
    // === APPROACH 1: Loss accumulation (our new approach) ===
    
    // Combine all logits and targets
    let combined_logits_data = vec![
        1.0, 2.0, 0.5, 0.8, 1.5, 2.1,  // batch 1
        2.5, 0.3, 0.7, 0.1, 0.4, 1.8,  // batch 2  
        1.2, 0.9, 1.1, 0.6, 0.2, 1.4,  // batch 3
    ];
    let combined_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(combined_logits_data, [6, 3]),
        &device
    ).require_grad();
    
    let combined_targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64, 0i64, 0i64, 2i64, 2i64, -1i64], [6]),
        &device
    );
    
    // Compute accumulated loss and gradients
    let accumulated_loss = compute_masked_cross_entropy_loss_ignore_index(
        combined_logits.clone(), 
        combined_targets, 
        &device
    );
    let accumulated_loss_value: f32 = accumulated_loss.clone().into_scalar();
    
    let accumulated_gradients = accumulated_loss.backward();
    let accumulated_grad = combined_logits.grad(&accumulated_gradients).unwrap();
    let accumulated_grad_data = accumulated_grad.clone().into_data();
    let accumulated_grad_values: Vec<f32> = accumulated_grad_data.iter::<f32>().collect();
    
    // === APPROACH 2: Individual losses summed (reference approach) ===
    
    // Compute individual losses
    let loss1 = compute_masked_cross_entropy_loss_ignore_index(logits1.clone(), targets1, &device);
    let loss1_value: f32 = loss1.clone().into_scalar();
    
    let loss2 = compute_masked_cross_entropy_loss_ignore_index(logits2.clone(), targets2, &device);
    let loss2_value: f32 = loss2.clone().into_scalar();
    
    let loss3 = compute_masked_cross_entropy_loss_ignore_index(logits3.clone(), targets3, &device);
    let loss3_value: f32 = loss3.clone().into_scalar();
    
    // Sum the individual losses
    let individual_loss_sum = loss1_value + loss2_value + loss3_value;
    
    // Compute individual gradients and sum them manually
    let grad1 = loss1.backward();
    let grad1_tensor = logits1.grad(&grad1).unwrap();
    let grad1_data = grad1_tensor.into_data();
    let grad1_values: Vec<f32> = grad1_data.iter::<f32>().collect();
    
    let grad2 = loss2.backward(); 
    let grad2_tensor = logits2.grad(&grad2).unwrap();
    let grad2_data = grad2_tensor.into_data();
    let grad2_values: Vec<f32> = grad2_data.iter::<f32>().collect();
    
    let grad3 = loss3.backward();
    let grad3_tensor = logits3.grad(&grad3).unwrap(); 
    let grad3_data = grad3_tensor.into_data();
    let grad3_values: Vec<f32> = grad3_data.iter::<f32>().collect();
    
    // Manually sum the gradients
    let mut summed_grad_values = Vec::new();
    for i in 0..6 {
        let batch_idx = i / 2;
        let elem_idx = i % 2;
        let grad_val = match batch_idx {
            0 => grad1_values[elem_idx * 3..(elem_idx + 1) * 3].to_vec(),
            1 => grad2_values[elem_idx * 3..(elem_idx + 1) * 3].to_vec(),
            2 => grad3_values[elem_idx * 3..(elem_idx + 1) * 3].to_vec(),
            _ => unreachable!(),
        };
        summed_grad_values.extend(grad_val);
    }
    
    println!("   Accumulated loss: {:.6}", accumulated_loss_value);
    println!("   Individual loss sum: {:.6}", individual_loss_sum);
    println!("   Loss difference: {:.8}", (accumulated_loss_value - individual_loss_sum).abs());
    
    // === VERIFICATION ===
    
    // 1. Loss values should be very close
    let loss_tolerance = 1e-5;
    let loss_diff = (accumulated_loss_value - individual_loss_sum).abs();
    assert!(loss_diff < loss_tolerance,
            "Accumulated loss should equal sum of individual losses. \
            Accumulated: {:.6}, Sum: {:.6}, Diff: {:.8}",
            accumulated_loss_value, individual_loss_sum, loss_diff);
    
    // 2. Gradients should be very close
    let grad_tolerance = 1e-5;
    assert_eq!(accumulated_grad_values.len(), summed_grad_values.len(),
               "Gradient arrays should have same length");
    
    let mut max_grad_diff: f32 = 0.0;
    let mut total_grad_diff = 0.0;
    for (i, (&acc_grad, &sum_grad)) in accumulated_grad_values.iter()
        .zip(summed_grad_values.iter()).enumerate() {
        let diff = (acc_grad - sum_grad).abs();
        max_grad_diff = max_grad_diff.max(diff);
        total_grad_diff += diff;
        
        if diff >= grad_tolerance {
            println!("   Large gradient difference at index {}: acc={:.8}, sum={:.8}, diff={:.8}",
                    i, acc_grad, sum_grad, diff);
        }
    }
    
    let avg_grad_diff = total_grad_diff / accumulated_grad_values.len() as f32;
    
    println!("   Max gradient difference: {:.8}", max_grad_diff);
    println!("   Average gradient difference: {:.8}", avg_grad_diff);
    
    assert!(max_grad_diff < grad_tolerance,
            "Maximum gradient difference should be small. Max diff: {:.8}",
            max_grad_diff);
    
    // 3. Both approaches should produce finite results
    assert!(accumulated_loss_value.is_finite(), "Accumulated loss should be finite");
    assert!(individual_loss_sum.is_finite(), "Individual loss sum should be finite");
    
    println!("✅ Test PASSED: Loss accumulation produces equivalent results to gradient accumulation");
}

/// Test — Gradient scaling consistency
/// 
/// Verify that when we scale the accumulated loss (e.g., for learning rate adjustment),
/// the gradients scale proportionally.
#[test]
fn test_gradient_scaling_consistency() {
    println!("Running Test: Gradient scaling consistency");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test data
    let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1, 2.5, 0.3, 0.7];
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data.clone(), [3, 3]),
        &device
    ).require_grad();
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64, 0i64, 2i64], [3]),
        &device
    );
    
    // Compute base loss and gradients
    let base_loss = compute_masked_cross_entropy_loss_ignore_index(logits.clone(), targets.clone(), &device);
    let base_loss_value: f32 = base_loss.clone().into_scalar();
    let base_grads = base_loss.backward();
    let base_grad_tensor = logits.grad(&base_grads).unwrap();
    let base_grad_data = base_grad_tensor.into_data();
    let base_grad_values: Vec<f32> = base_grad_data.iter::<f32>().collect();
    
    // Test with different scaling factors
    let scale_factors = vec![0.5, 2.0, 0.1, 10.0];
    
    for scale_factor in scale_factors {
        // Create fresh logits tensor for gradient computation
        let scaled_logits = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(logits_data.clone(), [3, 3]),
            &device
        ).require_grad();
        
        let scaled_loss = compute_masked_cross_entropy_loss_ignore_index(
            scaled_logits.clone(), 
            targets.clone(), 
            &device
        ) * scale_factor;
        
        let scaled_loss_value: f32 = scaled_loss.clone().into_scalar();
        let scaled_grads = scaled_loss.backward();
        let scaled_grad_tensor = scaled_logits.grad(&scaled_grads).unwrap();
        let scaled_grad_data = scaled_grad_tensor.into_data();
        let scaled_grad_values: Vec<f32> = scaled_grad_data.iter::<f32>().collect();
        
        // Verify loss scaling
        let expected_scaled_loss = base_loss_value * scale_factor;
        let loss_diff = (scaled_loss_value - expected_scaled_loss).abs();
        assert!(loss_diff < 1e-5, 
                "Scaled loss should equal base_loss * scale_factor. \
                Scale: {}, Expected: {:.6}, Got: {:.6}",
                scale_factor, expected_scaled_loss, scaled_loss_value);
        
        // Verify gradient scaling
        for (i, (&base_grad, &scaled_grad)) in base_grad_values.iter()
            .zip(scaled_grad_values.iter()).enumerate() {
            let expected_scaled_grad = base_grad * scale_factor;
            let grad_diff = (scaled_grad - expected_scaled_grad).abs();
            
            // Allow larger tolerance for very small gradients
            let tolerance = if expected_scaled_grad.abs() < 1e-6 { 1e-6 } else { 1e-5 };
            
            assert!(grad_diff < tolerance,
                    "Scaled gradient should equal base_grad * scale_factor. \
                    Index: {}, Scale: {}, Base: {:.8}, Expected: {:.8}, Got: {:.8}",
                    i, scale_factor, base_grad, expected_scaled_grad, scaled_grad);
        }
        
        println!("   ✅ Scale factor {}: loss and gradients scale correctly", scale_factor);
    }
    
    println!("✅ Test PASSED: Gradient scaling is consistent");
}

/// Test — Accumulation with masked samples
/// 
/// Verify that gradient accumulation works correctly when some batches
/// have all masked samples (should contribute zero to accumulated gradients).
#[test]
fn test_accumulation_with_masked_samples() {
    println!("Running Test: Accumulation with masked samples");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Batch 1: Normal samples
    let batch1_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1], [2, 3]),
        &device
    ).require_grad();
    let batch1_targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64, 0i64], [2]),
        &device
    );
    
    // Batch 2: All masked samples
    let batch2_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![2.5, 0.3, 0.7, 0.1, 0.4, 1.8], [2, 3]),
        &device
    ).require_grad();
    let batch2_targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![-1i64, -1i64], [2]), // All masked
        &device
    );
    
    // Batch 3: Normal samples
    let batch3_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.2, 0.9, 1.1, 0.6, 0.2, 1.4], [2, 3]),
        &device
    ).require_grad();
    let batch3_targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![2i64, 1i64], [2]),
        &device
    );
    
    // Compute individual losses
    let loss1 = compute_masked_cross_entropy_loss_ignore_index(batch1_logits.clone(), batch1_targets, &device);
    let loss2 = compute_masked_cross_entropy_loss_ignore_index(batch2_logits.clone(), batch2_targets, &device);
    let loss3 = compute_masked_cross_entropy_loss_ignore_index(batch3_logits.clone(), batch3_targets, &device);
    
    let loss1_value: f32 = loss1.clone().into_scalar();
    let loss2_value: f32 = loss2.clone().into_scalar();
    let loss3_value: f32 = loss3.clone().into_scalar();
    
    println!("   Batch 1 loss (normal): {:.6}", loss1_value);
    println!("   Batch 2 loss (all masked): {:.8}", loss2_value);
    println!("   Batch 3 loss (normal): {:.6}", loss3_value);
    
    // Verify that the all-masked batch has zero loss
    assert!(loss2_value.abs() < 1e-6, 
            "All-masked batch should have zero loss, got: {}", loss2_value);
    
    // Verify that normal batches have positive loss
    assert!(loss1_value > 0.0 && loss1_value.is_finite(), "Batch 1 should have positive finite loss");
    assert!(loss3_value > 0.0 && loss3_value.is_finite(), "Batch 3 should have positive finite loss");
    
    // The accumulated loss should equal loss1 + loss3 (batch 2 contributes 0)
    let expected_total = loss1_value + loss3_value;
    
    // Now test with combined batches
    let combined_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![
            1.0, 2.0, 0.5, 0.8, 1.5, 2.1,    // batch 1
            2.5, 0.3, 0.7, 0.1, 0.4, 1.8,    // batch 2 (masked)
            1.2, 0.9, 1.1, 0.6, 0.2, 1.4,    // batch 3
        ], [6, 3]),
        &device
    );
    let combined_targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64, 0i64, -1i64, -1i64, 2i64, 1i64], [6]),
        &device
    );
    
    let combined_loss = compute_masked_cross_entropy_loss_ignore_index(combined_logits, combined_targets, &device);
    let combined_loss_value: f32 = combined_loss.into_scalar();
    
    println!("   Expected total: {:.6}", expected_total);
    println!("   Combined loss: {:.6}", combined_loss_value);
    
    let tolerance = 1e-5;
    let diff = (combined_loss_value - expected_total).abs();
    assert!(diff < tolerance,
            "Combined loss should equal sum of non-masked batches. \
            Expected: {:.6}, Got: {:.6}, Diff: {:.8}",
            expected_total, combined_loss_value, diff);
    
    println!("✅ Test PASSED: Accumulation correctly handles masked samples");
}