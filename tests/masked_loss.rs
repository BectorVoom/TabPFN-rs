// tests/masked_loss.rs - High-priority masked cross-entropy tests
use burn::tensor::{backend::Backend, Tensor, TensorData, Int};
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::loss_utils::compute_masked_cross_entropy_loss;

type TestBackend = NdArray<f32>;

#[test]
fn test_masked_loss_basic_functionality() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test basic masked loss computation
    // logits: [batch=2, classes=3]
    let logits_data = vec![
        1.0, 2.0, 0.5,  // First example
        0.8, 1.5, 2.1,  // Second example  
    ];
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [6]),
        &device
    ).reshape([2, 3]);
    
    // targets: [batch=2], both valid class indices
    let targets_data = vec![1i64, 0i64]; // First target (class 1), second target (class 0)
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [2]),
        &device
    );
    
    // mask: [batch=2], first valid, second masked
    let mask_data = vec![true, false]; // First example valid, second example masked
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(mask_data, [2]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    let loss_value: f32 = loss.to_data().as_slice::<f32>().unwrap()[0];
    
    // Verify loss properties
    assert!(loss_value.is_finite(), "Loss should be finite");
    assert!(loss_value > 0.0, "Loss should be positive");
    
    // Loss should only account for the first example (target=1)
    // With logits [1.0, 2.0, 0.5], target=1 should give reasonable loss
}

#[test]  
fn test_masked_loss_no_masking() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test with no masked examples (all targets valid)
    let logits_data = vec![
        1.0, 0.0, 0.0,  // Should predict class 0
        0.0, 1.0, 0.0,  // Should predict class 1
        0.0, 0.0, 1.0,  // Should predict class 2
    ];
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [9]),
        &device
    ).reshape([3, 3]);
    
    let targets_data = vec![0i64, 1i64, 2i64]; // All targets valid
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [3]),
        &device
    );
    
    // mask: all true (no masking)
    let mask_data = vec![true, true, true]; // All examples valid
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(mask_data, [3]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    let loss_value: f32 = loss.to_data().as_slice::<f32>().unwrap()[0];
    
    // Loss should be relatively low since logits match targets
    assert!(loss_value.is_finite());
    assert!(loss_value > 0.0);
    assert!(loss_value < 2.0, "Loss should be reasonable for correct predictions");
}

#[test]
#[should_panic(expected = "Masked loss: no valid positions in mask")]
fn test_masked_loss_all_masked_panics() {
    let device = <TestBackend as Backend>::Device::default();
    
    let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1];
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [6]),
        &device
    ).reshape([2, 3]);
    
    // Use valid target indices (will be ignored due to mask)
    let targets_data = vec![0i64, 1i64];
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [2]),
        &device
    );
    
    // mask: all false (completely masked)
    let mask_data = vec![false, false]; // All examples masked
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(mask_data, [2]),
        &device
    );
    
    // Should panic with clear message
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
fn test_masked_loss_gradient_preservation() {
    use burn::backend::Autodiff;
    type AutodiffBackend = Autodiff<NdArray<f32>>;
    
    let device = <AutodiffBackend as Backend>::Device::default();
    
    // Test that gradients flow correctly through masked loss
    let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1];
    let logits = Tensor::<AutodiffBackend, 1>::from_data(
        TensorData::new(logits_data, [6]),
        &device
    ).reshape([2, 3]).require_grad();
    
    let targets_data = vec![1i64, 0i64]; // Valid target indices
    let targets = Tensor::<AutodiffBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [2]),
        &device
    );
    
    // mask: first valid, second masked
    let mask_data = vec![true, false]; // First example valid, second masked
    let mask = Tensor::<AutodiffBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(mask_data, [2]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss(logits.clone(), targets, mask, &device);
    
    // Compute gradients
    let gradients = loss.backward();
    let logits_grad = logits.grad(&gradients).unwrap();
    
    // Verify gradients exist and are reasonable
    let grad_data = logits_grad.to_data();
    let grad_values = grad_data.as_slice::<f32>().unwrap();
    
    // First row (valid target) should have non-zero gradients
    assert!(grad_values[0].abs() > 1e-6 || grad_values[1].abs() > 1e-6 || grad_values[2].abs() > 1e-6,
            "Valid example should have gradients");
    
    // All gradients should be finite
    for &grad in grad_values.iter() {
        assert!(grad.is_finite(), "Gradients should be finite");
    }
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits batch dimension 3 must match targets batch dimension 1")]
fn test_masked_loss_invalid_logits_shape() {
    let device = <TestBackend as Backend>::Device::default();
    
    // logits has batch dimension 3, targets has batch dimension 1 - mismatch
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]),
        &device
    ).reshape([3, 2]); // batch=3, classes=2
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64], [1]), // batch=1
        &device
    );
    
    // Dummy mask (will not be used due to shape error)
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(vec![true], [1]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits batch dimension 1 must match targets batch dimension 2")]
fn test_masked_loss_invalid_targets_shape() {
    let device = <TestBackend as Backend>::Device::default();
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0], [3]),
        &device
    ).reshape([1, 3]); // batch=1, classes=3
    
    // targets with mismatched batch dimension
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64], [2]), // batch=2
        &device
    );
    
    // Dummy mask (will not be used due to shape error)
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(vec![true], [1]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits batch dimension")]
fn test_masked_loss_batch_dimension_mismatch() {
    let device = <TestBackend as Backend>::Device::default();
    
    // logits: [2, 3] 
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]),
        &device
    ).reshape([2, 3]);
    
    // targets: [3] - batch dimension mismatch
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64, 2i64], [3]),
        &device
    );
    
    // Dummy mask (will not be used due to shape error)
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(vec![true, true, true], [3]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
fn test_masked_loss_mixed_masking_pattern() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test various masking patterns
    let logits_data = vec![
        1.0, 2.0, 0.0, // Example 1: should predict class 1
        0.0, 0.0, 3.0, // Example 2: should predict class 2  
        2.0, 1.0, 0.0, // Example 3: should predict class 0
        1.0, 1.0, 1.0, // Example 4: ambiguous
    ];
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [12]),
        &device
    ).reshape([4, 3]);
    
    // Mixed masking: valid, masked, valid, masked
    let targets_data = vec![1i64, 2i64, 0i64, 1i64]; // Use valid class indices
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [4]),
        &device
    );
    
    // mask: valid, masked, valid, masked
    let mask_data = vec![true, false, true, false]; // Examples 1 and 3 valid, 2 and 4 masked
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(mask_data, [4]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    let loss_value: f32 = loss.to_data().as_slice::<f32>().unwrap()[0];
    
    // Should only account for examples 1 and 3 
    assert!(loss_value.is_finite());
    assert!(loss_value > 0.0);
    // Loss should be reasonable since we have correct predictions
    assert!(loss_value < 5.0, "Loss should be reasonable for mixed valid examples");
}

#[test]
fn test_masked_loss_single_example() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test with single example
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![0.0, 1.0, 0.0], [3]),
        &device
    ).reshape([1, 3]);
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![1i64], [1]), // Correct class
        &device
    );
    
    // mask: single valid example
    let mask = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(vec![true], [1]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    let loss_value: f32 = loss.to_data().as_slice::<f32>().unwrap()[0];
    
    assert!(loss_value.is_finite());
    assert!(loss_value > 0.0);
    // Should be relatively low loss since prediction is correct
    assert!(loss_value < 2.0, "Loss should be low for correct single prediction");
}

#[test]
fn test_masked_loss_deterministic_behavior() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that same inputs produce same outputs (deterministic)
    let logits_data = vec![1.5, 2.5, 0.5, 0.8, 1.2, 2.8];
    let targets_data = vec![1i64, 2i64];
    
    // First computation
    let logits1 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data.clone(), [6]),
        &device
    ).reshape([2, 3]);
    let targets1 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data.clone(), [2]),
        &device
    );
    // mask: all valid examples
    let mask1 = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(vec![true, true], [2]),
        &device
    );
    let loss1 = compute_masked_cross_entropy_loss(logits1, targets1, mask1, &device);
    let loss_value1: f32 = loss1.to_data().as_slice::<f32>().unwrap()[0];
    
    // Second computation with same data
    let logits2 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [6]),
        &device
    ).reshape([2, 3]);
    let targets2 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [2]),
        &device
    );
    // mask: all valid examples
    let mask2 = Tensor::<TestBackend, 1, burn::prelude::Bool>::from_data(
        TensorData::new(vec![true, true], [2]),
        &device
    );
    let loss2 = compute_masked_cross_entropy_loss(logits2, targets2, mask2, &device);
    let loss_value2: f32 = loss2.to_data().as_slice::<f32>().unwrap()[0];
    
    // Results should be identical
    assert!((loss_value1 - loss_value2).abs() < 1e-6, 
            "Masked loss should be deterministic: {} vs {}", loss_value1, loss_value2);
}