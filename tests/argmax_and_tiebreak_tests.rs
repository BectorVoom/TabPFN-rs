/// Argmax and Tie-Breaking Tests
///
/// Test suite to verify that argmax_with_tie_break_smallest function correctly handles:
/// 1. Deterministic tie-breaking using smallest index rule
/// 2. Correct input/output shapes: [S,B,C] → [S,B]
/// 3. Edge cases: all equal logits, single class, extreme values
/// 4. Input shape validation and error handling

use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_ndarray::{NdArray};

use tab_pfn_rs::{
    tabpfn::architectures::base::train::argmax_with_tie_break_smallest,
};

type TestBackend = NdArray<f32>;

/// Test basic argmax with deterministic tie-breaking
#[test]
fn test_argmax_deterministic_tie_breaking() {
    println!("Testing argmax with deterministic tie-breaking - smallest index wins");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits with known tie patterns
    // Shape: [S=2, B=3, C=4]
    let logits_data = vec![
        // Sequence 0, Batch 0: [1.0, 1.0, 0.5, 0.3] - tie between class 0 and 1, expect class 0
        1.0, 1.0, 0.5, 0.3,
        // Sequence 0, Batch 1: [0.8, 0.8, 0.8, 0.2] - 3-way tie, expect class 0  
        0.8, 0.8, 0.8, 0.2,
        // Sequence 0, Batch 2: [0.1, 0.9, 0.9, 0.1] - tie between class 1 and 2, expect class 1
        0.1, 0.9, 0.9, 0.1,
        
        // Sequence 1, Batch 0: [2.0, 1.0, 2.0, 1.5] - tie between class 0 and 2, expect class 0
        2.0, 1.0, 2.0, 1.5,
        // Sequence 1, Batch 1: [0.5, 0.5, 0.5, 0.5] - 4-way tie, expect class 0
        0.5, 0.5, 0.5, 0.5,
        // Sequence 1, Batch 2: [1.1, 1.2, 1.2, 1.0] - tie between class 1 and 2, expect class 1
        1.1, 1.2, 1.2, 1.0,
    ];
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [2 * 3 * 4]),
        &device
    ).reshape([2, 3, 4]); // [S=2, B=3, C=4]
    
    // Call argmax function
    let result = argmax_with_tie_break_smallest(logits);
    
    // Verify output shape
    assert_eq!(
        result.dims(),
        [2, 3],
        "Argmax result shape mismatch: expected [S=2, B=3], got {:?}",
        result.dims()
    );
    
    // Extract result data for validation
    let result_data = result.to_data();
    let result_slice = result_data.as_slice::<i64>().expect("Result should be i64 tensor");
    
    // Expected results based on tie-breaking rule (smallest index wins)
    let expected = vec![
        0i64, 0i64, 1i64,  // Sequence 0: [0, 0, 1] 
        0i64, 0i64, 1i64,  // Sequence 1: [0, 0, 1]
    ];
    
    assert_eq!(
        result_slice,
        expected.as_slice(),
        "Argmax tie-breaking failed. Expected {:?}, got {:?}. Tie-breaking must select smallest index.",
        expected, result_slice
    );
    
    println!("✅ Deterministic tie-breaking test PASSED: smallest indices selected correctly");
}

/// Test argmax with no ties (clear maximum values)
#[test]
fn test_argmax_clear_maxima() {
    println!("Testing argmax with clear maximum values (no ties)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits with clear maxima - shape [S=3, B=2, C=3]
    let logits_data = vec![
        // Sequence 0, Batch 0: [0.1, 0.9, 0.3] - max at class 1
        0.1, 0.9, 0.3,
        // Sequence 0, Batch 1: [2.1, 0.5, 1.2] - max at class 0
        2.1, 0.5, 1.2,
        
        // Sequence 1, Batch 0: [0.8, 0.2, 1.5] - max at class 2
        0.8, 0.2, 1.5, 
        // Sequence 1, Batch 1: [1.0, 2.0, 0.7] - max at class 1
        1.0, 2.0, 0.7,
        
        // Sequence 2, Batch 0: [3.0, 1.1, 2.9] - max at class 0
        3.0, 1.1, 2.9,
        // Sequence 2, Batch 1: [0.1, 0.2, 0.3] - max at class 2
        0.1, 0.2, 0.3,
    ];
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [3 * 2 * 3]),
        &device
    ).reshape([3, 2, 3]); // [S=3, B=2, C=3]
    
    let result = argmax_with_tie_break_smallest(logits);
    
    // Verify output shape
    assert_eq!(result.dims(), [3, 2], "Clear maxima result shape mismatch");
    
    let result_data = result.to_data();
    let result_slice = result_data.as_slice::<i64>().unwrap();
    
    let expected = vec![
        1i64, 0i64,  // Sequence 0: [1, 0]
        2i64, 1i64,  // Sequence 1: [2, 1] 
        0i64, 2i64,  // Sequence 2: [0, 2]
    ];
    
    assert_eq!(
        result_slice, expected.as_slice(),
        "Clear maxima argmax failed. Expected {:?}, got {:?}",
        expected, result_slice
    );
    
    println!("✅ Clear maxima test PASSED: correct maximum indices selected");
}

/// Test argmax with extreme values (very large and very small)
#[test]
fn test_argmax_extreme_values() {
    println!("Testing argmax with extreme values");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits with extreme values - shape [S=2, B=2, C=3]
    let logits_data = vec![
        // Sequence 0, Batch 0: [1000.0, -1000.0, 0.0] - max at class 0
        1000.0, -1000.0, 0.0,
        // Sequence 0, Batch 1: [-999.0, -999.0, 999.0] - max at class 2
        -999.0, -999.0, 999.0,
        
        // Sequence 1, Batch 0: [f32::INFINITY, 500.0, -f32::INFINITY] - max at class 0
        f32::INFINITY, 500.0, -f32::INFINITY,
        // Sequence 1, Batch 1: [0.0, f32::INFINITY, f32::INFINITY] - tie at infinity, expect class 1
        0.0, f32::INFINITY, f32::INFINITY,
    ];
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [2 * 2 * 3]),
        &device
    ).reshape([2, 2, 3]); // [S=2, B=2, C=3]
    
    let result = argmax_with_tie_break_smallest(logits);
    
    assert_eq!(result.dims(), [2, 2], "Extreme values result shape mismatch");
    
    let result_data = result.to_data();
    let result_slice = result_data.as_slice::<i64>().unwrap();
    
    let expected = vec![
        0i64, 2i64,  // Sequence 0: [0, 2]  
        0i64, 1i64,  // Sequence 1: [0, 1] - infinity tie broken by smallest index
    ];
    
    assert_eq!(
        result_slice, expected.as_slice(),
        "Extreme values argmax failed. Expected {:?}, got {:?}",
        expected, result_slice
    );
    
    println!("✅ Extreme values test PASSED: handles infinity and large values correctly");
}

/// Test argmax with single class (C=1)
#[test]
fn test_argmax_single_class() {
    println!("Testing argmax with single class");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits with single class - shape [S=3, B=2, C=1]
    let logits_data = vec![
        5.0,   // Sequence 0, Batch 0
        -2.0,  // Sequence 0, Batch 1
        0.0,   // Sequence 1, Batch 0
        100.0, // Sequence 1, Batch 1  
        -50.0, // Sequence 2, Batch 0
        0.001, // Sequence 2, Batch 1
    ];
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [3 * 2 * 1]),
        &device
    ).reshape([3, 2, 1]); // [S=3, B=2, C=1]
    
    let result = argmax_with_tie_break_smallest(logits);
    
    assert_eq!(result.dims(), [3, 2], "Single class result shape mismatch");
    
    let result_data = result.to_data();
    let result_slice = result_data.as_slice::<i64>().unwrap();
    
    // With single class, all results should be 0
    let expected = vec![0i64; 6]; // All zeros for [S=3, B=2]
    
    assert_eq!(
        result_slice, expected.as_slice(),
        "Single class argmax failed. All results should be 0, got {:?}",
        result_slice
    );
    
    println!("✅ Single class test PASSED: all results are class 0");
}

/// Test that argmax function works correctly with minimum valid dimensions
#[test]
fn test_argmax_minimal_dimensions() {
    println!("Testing argmax with minimal valid dimensions [S=1, B=1, C=1]");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create minimal 3D tensor [S=1, B=1, C=1]
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![5.0], [1]),
        &device
    ).reshape([1, 1, 1]); // [S=1, B=1, C=1]
    
    let result = argmax_with_tie_break_smallest(logits);
    
    assert_eq!(result.dims(), [1, 1], "Minimal dimensions result shape mismatch");
    
    let result_data = result.to_data();
    let result_slice = result_data.as_slice::<i64>().unwrap();
    
    // With single element, result should be class 0
    assert_eq!(result_slice, &[0i64], "Minimal dimensions should return class 0");
    
    println!("✅ Minimal dimensions test PASSED: [1,1,1] → [1,1] with class 0");
}

/// Test argmax with all equal values (complete tie scenario)
#[test]
fn test_argmax_all_equal_complete_tie() {
    println!("Testing argmax with all equal values (complete tie)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits where all values are equal - shape [S=2, B=2, C=4] 
    let equal_value = 1.5f32;
    let logits_data = vec![equal_value; 2 * 2 * 4]; // All values equal
    
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [2 * 2 * 4]),
        &device
    ).reshape([2, 2, 4]); // [S=2, B=2, C=4]
    
    let result = argmax_with_tie_break_smallest(logits);
    
    assert_eq!(result.dims(), [2, 2], "Complete tie result shape mismatch");
    
    let result_data = result.to_data();
    let result_slice = result_data.as_slice::<i64>().unwrap();
    
    // With complete ties, all results should be 0 (smallest index)
    let expected = vec![0i64; 4]; // All zeros for [S=2, B=2]
    
    assert_eq!(
        result_slice, expected.as_slice(), 
        "Complete tie argmax failed. All results should be 0 (smallest index), got {:?}",
        result_slice
    );
    
    println!("✅ Complete tie test PASSED: smallest index (0) selected in all cases");
}

/// Test argmax data type correctness
#[test]
fn test_argmax_output_data_type() {
    println!("Testing argmax output data type");
    
    let device = <TestBackend as Backend>::Device::default();
    
    let logits_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let logits = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [6]),
        &device
    ).reshape([2, 1, 3]); // [S=2, B=1, C=3]
    
    let result = argmax_with_tie_break_smallest(logits);
    
    // Verify the result can be converted to i64 (class indices)
    let result_data = result.to_data();
    assert!(
        result_data.as_slice::<i64>().is_ok(),
        "Argmax result must be i64 type for class indices"
    );
    
    // Verify values are valid class indices
    let result_slice = result_data.as_slice::<i64>().unwrap();
    for &class_idx in result_slice {
        assert!(
            class_idx >= 0 && class_idx < 3,
            "Class index {} is outside valid range [0, 3)",
            class_idx
        );
    }
    
    println!("✅ Output data type test PASSED: i64 type with valid class indices");
}