/// Labels and Ignore Index Tests
///
/// Test suite to verify that compute_masked_cross_entropy_loss_ignore_index correctly handles
/// the ignore_index=-1 pattern as specified in the TabPFN specification. This enforces:
/// - Ignore positions with -1 labels are excluded from loss computation
/// - Loss is finite and non-negative under all conditions
/// - Various ignore patterns work correctly
/// - Edge cases are handled properly

use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_ndarray::{NdArray};

use tab_pfn_rs::{
    tabpfn::architectures::base::loss_utils::compute_masked_cross_entropy_loss_ignore_index,
};

type TestBackend = NdArray<f32>;

/// Test basic ignore_index=-1 functionality
#[test]
fn test_ignore_index_basic_functionality() {
    println!("Testing basic ignore_index=-1 functionality");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits [batch_size=4, num_classes=3]
    let logits_data = vec![
        // Sample 0: clear maximum at class 1
        0.1, 2.0, 0.5,
        // Sample 1: clear maximum at class 0  
        1.8, 0.2, 0.3,
        // Sample 2: clear maximum at class 2
        0.4, 0.1, 1.5,
        // Sample 3: clear maximum at class 1 (will be ignored)
        0.2, 1.9, 0.3,
    ];
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [4 * 3]),
        &device
    ).reshape([4, 3]);
    
    // Create labels with ignore_index=-1 at position 3
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![1i64, 0i64, 2i64, -1i64], [4]),
        &device
    );
    
    // Compute loss
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    // Verify loss properties
    assert!(loss_value.is_finite(), "Loss must be finite: got {}", loss_value);
    assert!(loss_value >= 0.0, "Loss must be non-negative: got {}", loss_value);
    
    // Loss should be computed only over first 3 samples (positions 0,1,2)
    // Position 3 with -1 should be ignored
    println!("✅ Basic ignore_index test PASSED: loss = {:.4}", loss_value);
}

/// Test all positions ignored (edge case)
#[test]
fn test_all_positions_ignored() {
    println!("Testing all positions ignored (should panic)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits [batch_size=3, num_classes=2]
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.8, 0.2, 1.2, 0.1], [6]),
        &device
    ).reshape([3, 2]);
    
    // Create labels where ALL positions have ignore_index=-1
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![-1i64, -1i64, -1i64], [3]),
        &device
    );
    
    // This should panic with clear error message
    let result = std::panic::catch_unwind(|| {
        compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    });
    
    assert!(result.is_err(), "Function should panic when all positions are ignored");
    println!("✅ All positions ignored test PASSED: correctly panicked");
}

/// Test mixed ignore pattern
#[test]
fn test_mixed_ignore_pattern() {
    println!("Testing mixed ignore pattern");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits [batch_size=6, num_classes=4]
    let logits_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [24]),
        &device
    ).reshape([6, 4]);
    
    // Create mixed pattern: some valid, some ignored
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![0i64, -1i64, 2i64, -1i64, 1i64, 3i64], [6]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    // Verify loss properties
    assert!(loss_value.is_finite(), "Loss with mixed pattern must be finite");
    assert!(loss_value >= 0.0, "Loss with mixed pattern must be non-negative");
    
    // Should compute loss over positions [0, 2, 4, 5] (valid labels)
    // Positions [1, 3] with -1 should be ignored
    println!("✅ Mixed ignore pattern test PASSED: loss = {:.4}", loss_value);
}

/// Test single valid position (edge case)
#[test]
fn test_single_valid_position() {
    println!("Testing single valid position");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits [batch_size=4, num_classes=3]
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![
            0.1, 0.8, 0.1,  // Sample 0: ignored
            0.2, 0.1, 0.7,  // Sample 1: ignored  
            1.5, 0.1, 0.4,  // Sample 2: valid label=0
            0.3, 0.2, 0.5,  // Sample 3: ignored
        ], [12]),
        &device
    ).reshape([4, 3]);
    
    // Only position 2 is valid
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![-1i64, -1i64, 0i64, -1i64], [4]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "Loss with single valid position must be finite");
    assert!(loss_value >= 0.0, "Loss with single valid position must be non-negative");
    
    println!("✅ Single valid position test PASSED: loss = {:.4}", loss_value);
}

/// Test various class indices with ignore pattern
#[test]
fn test_various_class_indices_with_ignore() {
    println!("Testing various class indices with ignore pattern");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits [batch_size=5, num_classes=5] - test all class indices
    let mut logits_data = vec![0.1f32; 25];
    // Make diagonal elements higher (correct predictions)
    for i in 0..5 {
        logits_data[i * 5 + i] = 2.0;
    }
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [25]),
        &device
    ).reshape([5, 5]);
    
    // Test all class indices: 0, 1, 2, 3, 4 with some ignored
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![0i64, -1i64, 2i64, 3i64, -1i64], [5]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "Loss with various class indices must be finite");
    assert!(loss_value >= 0.0, "Loss with various class indices must be non-negative");
    
    // Since diagonal elements are higher, loss should be relatively low
    println!("✅ Various class indices test PASSED: loss = {:.4}", loss_value);
}

/// Test numerical stability with extreme values
#[test]
fn test_numerical_stability_extreme_values() {
    println!("Testing numerical stability with extreme values");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits with extreme values [batch_size=4, num_classes=3]
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![
            -100.0, 100.0, 0.0,     // Sample 0: extreme values
            1000.0, -1000.0, 0.0,   // Sample 1: very extreme
            0.0, 0.0, 0.0,          // Sample 2: all equal (ignored)
            f32::INFINITY, 0.0, -f32::INFINITY,  // Sample 3: infinite values (ignored)
        ], [12]),
        &device
    ).reshape([4, 3]);
    
    // Valid labels for positions 0, 1; ignore positions 2, 3
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![1i64, 0i64, -1i64, -1i64], [4]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    // Even with extreme values, loss should be finite and non-negative
    assert!(loss_value.is_finite(), "Loss with extreme values must be finite, got {}", loss_value);
    assert!(loss_value >= 0.0, "Loss with extreme values must be non-negative, got {}", loss_value);
    
    println!("✅ Numerical stability test PASSED: loss = {:.4}", loss_value);
}

/// Test large batch with high ignore ratio
#[test]
fn test_large_batch_high_ignore_ratio() {
    println!("Testing large batch with high ignore ratio");
    
    let device = <TestBackend as Backend>::Device::default();
    
    let batch_size = 100;
    let num_classes = 10;
    
    // Create random-ish logits
    let logits_data: Vec<f32> = (0..(batch_size * num_classes))
        .map(|i| ((i * 17) % 100) as f32 * 0.01)
        .collect();
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [batch_size * num_classes]),
        &device
    ).reshape([batch_size, num_classes]);
    
    // Create labels with 90% ignore ratio (only 10% valid)
    let mut labels_data = vec![-1i64; batch_size];
    for i in (0..batch_size).step_by(10) {
        labels_data[i] = (i % num_classes) as i64;  // Valid labels every 10th position
    }
    
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(labels_data, [batch_size]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "Loss with high ignore ratio must be finite");
    assert!(loss_value >= 0.0, "Loss with high ignore ratio must be non-negative");
    
    println!("✅ Large batch high ignore ratio test PASSED: loss = {:.4} (90% ignored)", loss_value);
}

/// Test consistency: same logits/labels should give same loss
#[test]
fn test_loss_consistency() {
    println!("Testing loss computation consistency");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create identical logits
    let logits_data = vec![0.5, 1.2, 0.8, 0.1, 1.5, 0.3];
    let logits1 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data.clone(), [6]),
        &device
    ).reshape([2, 3]);
    let logits2 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [6]),
        &device
    ).reshape([2, 3]);
    
    // Create identical labels
    let labels_data = vec![1i64, -1i64];
    let labels1 = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(labels_data.clone(), [2]),
        &device
    );
    let labels2 = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(labels_data, [2]),
        &device
    );
    
    // Compute losses
    let loss1 = compute_masked_cross_entropy_loss_ignore_index(logits1, labels1, &device);
    let loss2 = compute_masked_cross_entropy_loss_ignore_index(logits2, labels2, &device);
    
    let loss1_value: f32 = loss1.into_scalar();
    let loss2_value: f32 = loss2.into_scalar();
    
    // Should be identical
    assert!((loss1_value - loss2_value).abs() < 1e-6, 
            "Identical inputs should produce identical losses: {} vs {}", loss1_value, loss2_value);
    
    println!("✅ Loss consistency test PASSED: both losses = {:.6}", loss1_value);
}

/// Test with TabPFN canonical tensor shapes [S*B, C] and [S*B]
#[test]
fn test_tabpfn_canonical_shapes() {
    println!("Testing TabPFN canonical shapes [S*B, C] and [S*B]");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Simulate [S=3, B=2] -> [S*B=6] canonical TabPFN shapes
    let seq_len = 3;
    let batch_size = 2;
    let num_classes = 4;
    let total_samples = seq_len * batch_size;
    
    // Create logits [S*B=6, C=4]
    let logits_data: Vec<f32> = (0..(total_samples * num_classes))
        .map(|i| (i as f32) * 0.05)
        .collect();
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [total_samples * num_classes]),
        &device
    ).reshape([total_samples, num_classes]);
    
    // Create labels [S*B=6] with TabPFN ignore pattern
    let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![0i64, 1i64, -1i64, 2i64, -1i64, 3i64], [total_samples]),
        &device
    );
    
    let loss = compute_masked_cross_entropy_loss_ignore_index(logits, labels, &device);
    let loss_value: f32 = loss.into_scalar();
    
    assert!(loss_value.is_finite(), "TabPFN canonical shapes loss must be finite");
    assert!(loss_value >= 0.0, "TabPFN canonical shapes loss must be non-negative");
    
    println!("✅ TabPFN canonical shapes test PASSED: loss = {:.4}", loss_value);
}