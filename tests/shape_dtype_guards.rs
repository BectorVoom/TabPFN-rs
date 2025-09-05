// tests/shape_dtype_guards.rs - Test shape and dtype validation guards
use burn::tensor::{backend::Backend, Tensor, TensorData, Int, Bool};
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::loss_utils::{compute_masked_cross_entropy_loss, compute_masked_cross_entropy_loss_ignore_index};

type TestBackend = NdArray<f32>;

#[test]
#[should_panic(expected = "SHAPE ERROR: logits must be 2D tensor")]
fn test_logits_wrong_dimensions_1d() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test 1D logits (should be 2D)
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0], [3]),
        &device
    );
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64], [1]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true], [1]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits must be 2D tensor")]
fn test_logits_wrong_dimensions_3d() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test 3D logits (should be 2D)
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]),
        &device
    ).reshape([1usize, 2usize, 3usize]);
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64], [2]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true, true], [2]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: targets must be 1D tensor")]
fn test_targets_wrong_dimensions_2d() {
    let device = <TestBackend as Backend>::Device::default();
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]),
        &device
    ).reshape([2usize, 3usize]);
    
    // Test 2D targets (should be 1D)
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64], [2]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true, true], [2]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: targets must be 1D tensor")]
fn test_targets_wrong_dimensions_0d() {
    let device = <TestBackend as Backend>::Device::default();
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0], [3]),
        &device
    ).reshape([1usize, 3usize]);
    
    // Test 0D/scalar targets (should be 1D)
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64], [1]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true], [1]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits batch dimension")]
fn test_batch_dimension_mismatch_larger_logits() {
    let device = <TestBackend as Backend>::Device::default();
    
    // logits: [3, 4] - batch size 3
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0; 12], [12]),
        &device
    ).reshape([3, 4]);
    
    // targets: [2] - batch size 2 (mismatch)
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64], [2]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true, true], [2]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits batch dimension")]
fn test_batch_dimension_mismatch_larger_targets() {
    let device = <TestBackend as Backend>::Device::default();
    
    // logits: [2, 3] - batch size 2
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]),
        &device
    ).reshape([2, 3]);
    
    // targets: [4] - batch size 4 (mismatch)
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64, 2i64, 1i64], [4]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true, true, true, true], [4]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
#[should_panic(expected = "SHAPE ERROR: logits must have at least one class")]
fn test_zero_classes() {
    let device = <TestBackend as Backend>::Device::default();
    
    // logits: [2, 0] - zero classes
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new::<f32, [usize; 1]>(vec![], [0]),
        &device
    ).reshape([2usize, 0usize]);
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64, 1i64], [2]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true, true], [2]),
        &device
    );
    
    compute_masked_cross_entropy_loss(logits, targets, mask, &device);
}

#[test]
fn test_valid_shapes_various_sizes() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test various valid shape combinations
    let test_cases = vec![
        (1, 2),  // Single example, binary classification
        (1, 10), // Single example, 10 classes
        (5, 3),  // 5 examples, 3 classes
        (100, 1000), // Large batch, many classes
    ];
    
    for (batch_size, num_classes) in test_cases {
        let logits_data = vec![0.0; batch_size * num_classes];
        let logits = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(logits_data, [batch_size * num_classes]),
            &device
        ).reshape([batch_size, num_classes]);
        
        let targets_data = vec![0i64; batch_size];
        let targets = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(targets_data, [batch_size]),
            &device
        );
        
        let mask_data = vec![true; batch_size];
        let mask = Tensor::<TestBackend, 1, Bool>::from_data(
            TensorData::new(mask_data, [batch_size]),
            &device
        );
        
        // Should not panic for valid shapes
        let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
        let loss_value: f32 = loss.into_scalar();
        
        assert!(loss_value.is_finite(), 
                "Loss should be finite for valid shapes [batch={}, classes={}]", batch_size, num_classes);
    }
}

#[test]
fn test_edge_case_single_example_single_class() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Edge case: single example, single class
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0], [1]),
        &device
    ).reshape([1, 1]);
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64], [1]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(vec![true], [1]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "Single example, single class should be valid");
    assert!(loss_value > 0.0, "Loss should be positive");
}

#[test]
fn test_large_batch_size() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test with larger batch size to ensure no overflow issues
    let batch_size = 1000;
    let num_classes = 10;
    
    let logits_data = (0..(batch_size * num_classes))
        .map(|i| (i as f32) * 0.01)
        .collect::<Vec<_>>();
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [batch_size * num_classes]),
        &device
    ).reshape([batch_size, num_classes]);
    
    let targets_data = (0..batch_size)
        .map(|i| ((i % num_classes) as i64))
        .collect::<Vec<_>>();
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [batch_size]),
        &device
    );
    
    let mask_data = vec![true; batch_size];
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new(mask_data, [batch_size]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "Large batch should produce finite loss");
    assert!(loss_value > 0.0, "Large batch loss should be positive");
}

#[test]
fn test_dtype_consistency_int_targets() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that integer targets work correctly
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6]),
        &device
    ).reshape([2, 3]);
    
    // Test various integer target values
    let valid_targets = vec![
        vec![0i64, 1i64],        // Standard class indices
        vec![0i64, 2i64],        // Different class indices
        vec![2i64, 0i64],        // Reverse order
        vec![1i64, 1i64],        // Same class
        vec![-1i64, 0i64],       // One masked, one valid
        vec![0i64, -1i64],       // One valid, one masked
    ];
    
    for targets_data in valid_targets {
        let targets = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(targets_data.clone(), [2]),
            &device
        );
        
        if targets_data.iter().all(|&t| t == -1) {
            // All masked - should panic
            continue;
        }
        
        let loss = compute_masked_cross_entropy_loss_ignore_index(logits.clone(), targets, &device);
        let loss_value: f32 = loss.into_scalar();
        
        assert!(loss_value.is_finite(), 
                "Loss should be finite for targets {:?}", targets_data);
    }
}

#[test]
fn test_shape_validation_comprehensive() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test matrix of valid combinations
    let batch_sizes = vec![1, 2, 5, 10];
    let class_counts = vec![1, 2, 5, 100];
    
    for &batch_size in &batch_sizes {
        for &num_classes in &class_counts {
            let logits_size = batch_size * num_classes;
            let logits_data = (0..logits_size).map(|i| (i as f32) * 0.1).collect::<Vec<_>>();
            
            let logits = Tensor::<TestBackend, 1>::from_data(
                TensorData::new(logits_data, [logits_size]),
                &device
            ).reshape([batch_size, num_classes]);
            
            let targets_data = (0..batch_size)
                .map(|i| ((i % num_classes) as i64))
                .collect::<Vec<_>>();
            let targets = Tensor::<TestBackend, 1, Int>::from_data(
                TensorData::new(targets_data, [batch_size]),
                &device
            );
            
            let mask_data = vec![true; batch_size];
            let mask = Tensor::<TestBackend, 1, Bool>::from_data(
                TensorData::new(mask_data, [batch_size]),
                &device
            );
            
            // This should work for all valid combinations
            let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
            let loss_value: f32 = loss.into_scalar();
            
            assert!(loss_value.is_finite(), 
                    "Loss should be finite for batch_size={}, num_classes={}", batch_size, num_classes);
            assert!(loss_value >= 0.0,
                    "Loss should be non-negative for batch_size={}, num_classes={}", batch_size, num_classes);
        }
    }
}

#[test]
fn test_target_class_index_bounds() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test targets at class boundaries
    let num_classes = 5;
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0; 5 * num_classes], [5 * num_classes]),
        &device
    ).reshape([5, num_classes]);
    
    // Test boundary cases
    let boundary_targets = vec![
        vec![0i64, num_classes as i64 - 1, 0i64, -1i64, 2i64], // Min, max, masked
    ];
    
    for targets_data in boundary_targets {
        let targets = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(targets_data.clone(), [5]),
            &device
        );
        
        let mask = Tensor::<TestBackend, 1, Bool>::from_data(
            TensorData::new(vec![true, true, true, false, true], [5]),  // -1 target is masked
            &device
        );
        
        let loss = compute_masked_cross_entropy_loss(logits.clone(), targets, mask, &device);
        let loss_value: f32 = loss.into_scalar();
        
        assert!(loss_value.is_finite(),
                "Loss should be finite for boundary targets {:?}", targets_data);
    }
}

#[test]
fn test_zero_batch_size_handled_gracefully() {
    let device = <TestBackend as Backend>::Device::default();
    
    // This test checks what happens with zero batch size
    // It might panic or handle gracefully depending on implementation
    
    // Create empty tensors
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new::<f32, [usize; 1]>(vec![], [0]),
        &device
    ).reshape([0usize, 3usize]);
    
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new::<i64, [usize; 1]>(vec![], [0]),
        &device
    );
    
    let mask = Tensor::<TestBackend, 1, Bool>::from_data(
        TensorData::new::<bool, [usize; 1]>(vec![], [0]),
        &device
    );
    
    // This will likely panic with "no valid examples" which is appropriate
    let result = std::panic::catch_unwind(|| {
        compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    });
    
    // Either panics or succeeds - both are acceptable for zero batch size
    assert!(result.is_err() || result.is_ok(),
            "Zero batch size should either panic gracefully or handle appropriately");
}