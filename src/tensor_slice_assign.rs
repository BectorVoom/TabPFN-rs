//! Tensor slice assignment implementation using Burn's slice_assign method
//! 
//! This module provides a safe, validated wrapper around Burn's tensor slice assignment
//! functionality with comprehensive error handling and shape validation.

use burn::prelude::*;
use std::ops::Range;

/// Updates a specific slice of a tensor with new values using Burn's slice_assign method.
/// 
/// This function performs comprehensive validation before calling slice_assign to ensure:
/// - Shape consistency between the slice region and values tensor
/// - Bounds checking for all ranges
/// - Proper error handling with descriptive messages
/// 
/// # Arguments
/// 
/// * `tensor` - The input tensor to update (any 4D shape [B, C, H, W])
/// * `values` - The values to assign to the slice region (must match slice shape exactly)
/// * `ranges` - Array of ranges defining the slice region for each dimension
/// 
/// # Returns
/// 
/// * `Ok(Tensor<B, 4>)` - New tensor with updated slice values
/// * `Err(String)` - Descriptive error message for validation failures
/// 
/// # Examples
/// 
/// ```rust
/// use burn::prelude::*;
/// use burn_ndarray::NdArray;
/// 
/// type Backend = NdArray<f32>;
/// let device = Default::default();
/// 
/// // Create a 2x3x8x8 tensor of zeros
/// let tensor = Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device);
/// 
/// // Create values to assign (2x3x4x4 ones)
/// let values = Tensor::<Backend, 4>::ones([2, 3, 4, 4], &device);
/// 
/// // Update slice [0..2, 0..3, 2..6, 2..6] with values
/// let ranges = [0..2, 0..3, 2..6, 2..6];
/// let result = update_tensor_slice_4d(tensor, values, ranges)?;
/// ```
pub fn update_tensor_slice_4d<B: Backend>(
    tensor: Tensor<B, 4>,
    values: Tensor<B, 4>,
    ranges: [Range<usize>; 4],
) -> Result<Tensor<B, 4>, String> {
    // Validate ranges and compute expected slice shape
    let tensor_shape = tensor.dims();
    let mut slice_shape = [0usize; 4];
    
    for (dim_idx, (range, &tensor_dim)) in ranges.iter().zip(tensor_shape.iter()).enumerate() {
        // Check for valid range (start < end)
        if range.start >= range.end {
            return Err(format!(
                "Invalid range: start {} >= end {} for dimension {}",
                range.start, range.end, dim_idx
            ));
        }
        
        // Check bounds
        if range.end > tensor_dim {
            return Err(format!(
                "Range out of bounds: dimension {} range {}..{} exceeds size {}",
                dim_idx, range.start, range.end, tensor_dim
            ));
        }
        
        slice_shape[dim_idx] = range.end - range.start;
    }
    
    // Validate values tensor shape matches slice shape
    let values_shape = values.dims();
    if slice_shape != values_shape {
        return Err(format!(
            "Shape mismatch: expected {:?}, got {:?}",
            slice_shape, values_shape
        ));
    }
    
    // Perform the slice assignment
    let result = tensor.slice_assign(ranges, values);
    Ok(result)
}

/// Generic version that works with any dimension count
pub fn update_tensor_slice<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    values: Tensor<B, D>,
    ranges: [Range<usize>; D],
) -> Result<Tensor<B, D>, String> {
    // Validate ranges and compute expected slice shape
    let tensor_shape = tensor.dims();
    let mut slice_shape = [0usize; D];
    
    for (dim_idx, (range, &tensor_dim)) in ranges.iter().zip(tensor_shape.iter()).enumerate() {
        // Check for valid range (start < end)
        if range.start >= range.end {
            return Err(format!(
                "Invalid range: start {} >= end {} for dimension {}",
                range.start, range.end, dim_idx
            ));
        }
        
        // Check bounds
        if range.end > tensor_dim {
            return Err(format!(
                "Range out of bounds: dimension {} range {}..{} exceeds size {}",
                dim_idx, range.start, range.end, tensor_dim
            ));
        }
        
        slice_shape[dim_idx] = range.end - range.start;
    }
    
    // Validate values tensor shape matches slice shape
    let values_shape = values.dims();
    if slice_shape != values_shape {
        return Err(format!(
            "Shape mismatch: expected {:?}, got {:?}",
            slice_shape, values_shape
        ));
    }
    
    // Perform the slice assignment
    let result = tensor.slice_assign(ranges, values);
    Ok(result)
}

/// Validates that a tensor slice region matches expected values
/// 
/// This helper function extracts a slice from a tensor and verifies all values
/// in that region match the expected value (within floating point tolerance).
pub fn validate_slice_values<B: Backend>(
    tensor: &Tensor<B, 4>,
    ranges: [Range<usize>; 4],
    expected_value: f32,
    tolerance: f32,
) -> bool {
    let slice = tensor.clone().slice(ranges);
    let slice_data = slice.to_data();
    let slice_values = slice_data.as_slice::<f32>().expect("Failed to convert to f32 slice");
    
    slice_values.iter().all(|&val| (val - expected_value).abs() <= tolerance)
}

/// Validates that regions outside the slice remain unchanged
pub fn validate_unchanged_regions<B: Backend>(
    original: &Tensor<B, 4>,
    updated: &Tensor<B, 4>,
    changed_ranges: [Range<usize>; 4],
    tolerance: f32,
) -> bool {
    let original_data = original.to_data();
    let updated_data = updated.to_data();
    let orig_values = original_data.as_slice::<f32>().expect("Failed to convert original to f32");
    let upd_values = updated_data.as_slice::<f32>().expect("Failed to convert updated to f32");
    
    let shape = original.dims();
    let [b_size, c_size, h_size, w_size] = shape;
    
    for b in 0..b_size {
        for c in 0..c_size {
            for h in 0..h_size {
                for w in 0..w_size {
                    // Check if this position is within the changed region
                    let in_changed_region = changed_ranges[0].contains(&b) &&
                                          changed_ranges[1].contains(&c) &&
                                          changed_ranges[2].contains(&h) &&
                                          changed_ranges[3].contains(&w);
                    
                    if !in_changed_region {
                        // This position should be unchanged
                        let linear_idx = b * c_size * h_size * w_size +
                                       c * h_size * w_size +
                                       h * w_size +
                                       w;
                        
                        if (orig_values[linear_idx] - upd_values[linear_idx]).abs() > tolerance {
                            return false;
                        }
                    }
                }
            }
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    // Helper function to create test tensors
    fn create_test_tensor(shape: [usize; 4], value: f32) -> Tensor<TestBackend, 4> {
        let device = Default::default();
        if value == 0.0 {
            Tensor::zeros(shape, &device)
        } else if value == 1.0 {
            Tensor::ones(shape, &device)
        } else {
            Tensor::ones(shape, &device) * value
        }
    }
    
    // Helper to assert error messages contain expected text
    fn assert_error_contains(result: &Result<Tensor<TestBackend, 4>, String>, expected_msg: &str) {
        match result {
            Err(msg) => assert!(
                msg.contains(expected_msg),
                "Error message '{}' does not contain '{}'", msg, expected_msg
            ),
            Ok(_) => panic!("Expected error but got Ok result"),
        }
    }
    
    #[test]
    fn test_basic_4d_slice_update() {
        // Test case A1: Basic 4D update
        let tensor = create_test_tensor([2, 3, 8, 8], 0.0);
        let values = create_test_tensor([2, 3, 4, 4], 1.0);
        let ranges = [0..2, 0..3, 2..6, 2..6];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify the slice region has value 1.0
        assert!(validate_slice_values(&result, ranges.clone(), 1.0, 1e-6));
        
        // Verify unchanged regions remain 0.0
        assert!(validate_unchanged_regions(&tensor, &result, ranges, 1e-6));
    }
    
    #[test]
    fn test_partial_channel_update() {
        // Test case A2: Partial channel update
        let tensor = create_test_tensor([1, 4, 6, 6], 1.0);
        let values = create_test_tensor([1, 2, 6, 6], 2.0);
        let ranges = [0..1, 1..3, 0..6, 0..6];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify the updated channels have value 2.0
        assert!(validate_slice_values(&result, ranges.clone(), 2.0, 1e-6));
        
        // Verify channels 0 and 3 remain 1.0
        let unchanged_ch0 = [0..1, 0..1, 0..6, 0..6];
        let unchanged_ch3 = [0..1, 3..4, 0..6, 0..6];
        assert!(validate_slice_values(&result, unchanged_ch0, 1.0, 1e-6));
        assert!(validate_slice_values(&result, unchanged_ch3, 1.0, 1e-6));
    }
    
    #[test]
    fn test_corner_slice_update() {
        // Test case A3: Corner slice
        let tensor = create_test_tensor([2, 2, 4, 4], 0.0);
        let values = create_test_tensor([1, 1, 2, 2], 3.0);
        let ranges = [0..1, 0..1, 0..2, 0..2];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify the corner region has value 3.0
        assert!(validate_slice_values(&result, ranges.clone(), 3.0, 1e-6));
        
        // Verify unchanged regions remain 0.0
        assert!(validate_unchanged_regions(&tensor, &result, ranges, 1e-6));
    }
    
    #[test]
    fn test_single_element_update() {
        // Test case B1: Single element
        let tensor = create_test_tensor([1, 1, 1, 1], 0.0);
        let values = create_test_tensor([1, 1, 1, 1], 5.0);
        let ranges = [0..1, 0..1, 0..1, 0..1];
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        
        // Verify the single element has value 5.0
        assert!(validate_slice_values(&result, ranges, 5.0, 1e-6));
    }
    
    #[test]
    fn test_full_tensor_update() {
        // Test case B2: Full tensor update
        let tensor = create_test_tensor([2, 2, 3, 3], 1.0);
        let values = create_test_tensor([2, 2, 3, 3], 7.0);
        let ranges = [0..2, 0..2, 0..3, 0..3];
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        
        // Verify entire tensor has value 7.0
        assert!(validate_slice_values(&result, ranges, 7.0, 1e-6));
    }
    
    #[test]
    fn test_boundary_slice_update() {
        // Test case B3: Boundary slices
        let tensor = create_test_tensor([2, 2, 4, 4], 0.0);
        let values = create_test_tensor([2, 2, 4, 2], 4.0);
        let ranges = [0..2, 0..2, 0..4, 2..4]; // Right edge
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify the boundary region has value 4.0
        assert!(validate_slice_values(&result, ranges.clone(), 4.0, 1e-6));
        
        // Verify unchanged regions remain 0.0
        assert!(validate_unchanged_regions(&tensor, &result, ranges, 1e-6));
    }
    
    #[test]
    fn test_shape_mismatch_error() {
        // Test case C1: Shape mismatch
        let tensor = create_test_tensor([2, 3, 8, 8], 0.0);
        let values = create_test_tensor([2, 3, 5, 5], 1.0); // Wrong shape
        let ranges = [0..2, 0..3, 2..6, 2..6]; // Expects [2,3,4,4]
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        assert_error_contains(&result, "Shape mismatch: expected [2, 3, 4, 4], got [2, 3, 5, 5]");
    }
    
    #[test]
    fn test_out_of_bounds_error() {
        // Test case C2: Out of bounds
        let tensor = create_test_tensor([2, 3, 8, 8], 0.0);
        let values = create_test_tensor([3, 5, 9, 9], 1.0);
        let ranges = [0..3, 0..5, 0..9, 0..9]; // Exceeds tensor size
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        assert_error_contains(&result, "Range out of bounds: dimension 0 range 0..3 exceeds size 2");
    }
    
    #[test]
    fn test_invalid_range_error() {
        // Test case C3: Invalid range
        let tensor = create_test_tensor([2, 3, 4, 4], 0.0);
        let values = create_test_tensor([1, 3, 4, 4], 1.0);
        let ranges = [2..1, 0..3, 0..4, 0..4]; // start > end
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        assert_error_contains(&result, "Invalid range: start 2 >= end 1 for dimension 0");
    }
}