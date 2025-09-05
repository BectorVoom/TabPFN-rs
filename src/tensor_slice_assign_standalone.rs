//! Standalone tensor slice assignment implementation for testing
//! 
//! This standalone version avoids dependencies on the transformer module
//! that has compilation issues.

use burn::prelude::*;
use std::ops::Range;

/// Updates a specific slice of a 4D tensor with new values using Burn's slice_assign method.
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

/// Validates that a tensor slice region matches expected values
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
    
    #[test]
    fn test_basic_slice_assignment() {
        // Create a 2x3x8x8 tensor of zeros
        let tensor = create_test_tensor([2, 3, 8, 8], 0.0);
        
        // Create values to assign (2x3x4x4 ones)
        let values = create_test_tensor([2, 3, 4, 4], 1.0);
        
        // Update slice [0..2, 0..3, 2..6, 2..6] with values
        let ranges = [0..2, 0..3, 2..6, 2..6];
        let result = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        
        // Verify the slice region has value 1.0
        assert!(validate_slice_values(&result, ranges, 1.0, 1e-6));
        
        // Verify corners remain 0.0
        let corner_ranges = [0..1, 0..1, 0..2, 0..2];
        assert!(validate_slice_values(&result, corner_ranges, 0.0, 1e-6));
    }
    
    #[test]
    fn test_shape_mismatch_error() {
        let tensor = create_test_tensor([2, 3, 8, 8], 0.0);
        let wrong_values = create_test_tensor([2, 3, 5, 5], 1.0); // Wrong shape
        let ranges = [0..2, 0..3, 2..6, 2..6]; // Expects [2,3,4,4]
        
        let result = update_tensor_slice_4d(tensor, wrong_values, ranges);
        
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Shape mismatch"));
        assert!(error_msg.contains("[2, 3, 4, 4]"));
        assert!(error_msg.contains("[2, 3, 5, 5]"));
    }
    
    #[test]
    fn test_out_of_bounds_error() {
        let tensor = create_test_tensor([2, 3, 8, 8], 0.0);
        let values = create_test_tensor([3, 5, 9, 9], 1.0);
        let ranges = [0..3, 0..5, 0..9, 0..9]; // Exceeds tensor size
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Range out of bounds"));
        assert!(error_msg.contains("dimension 0"));
        assert!(error_msg.contains("exceeds size 2"));
    }
    
    #[test]
    fn test_invalid_range_error() {
        let tensor = create_test_tensor([2, 3, 4, 4], 0.0);
        let values = create_test_tensor([1, 3, 4, 4], 1.0);
        let ranges = [2..1, 0..3, 0..4, 0..4]; // Invalid: start > end
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Invalid range"));
        assert!(error_msg.contains("start 2 >= end 1"));
        assert!(error_msg.contains("dimension 0"));
    }
    
    #[test]
    fn test_partial_channel_update() {
        // Test updating specific channels
        let tensor = create_test_tensor([1, 4, 6, 6], 1.0);
        let values = create_test_tensor([1, 2, 6, 6], 2.0);
        let ranges = [0..1, 1..3, 0..6, 0..6]; // Update channels 1-2
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        
        // Verify updated channels have value 2.0
        assert!(validate_slice_values(&result, ranges, 2.0, 1e-6));
        
        // Verify other channels remain 1.0
        let ch0_ranges = [0..1, 0..1, 0..6, 0..6];
        let ch3_ranges = [0..1, 3..4, 0..6, 0..6];
        assert!(validate_slice_values(&result, ch0_ranges, 1.0, 1e-6));
        assert!(validate_slice_values(&result, ch3_ranges, 1.0, 1e-6));
    }
    
    #[test]
    fn test_precision_preservation() {
        let device = Default::default();
        
        // Test with specific float values
        let tensor = Tensor::<TestBackend, 4>::ones([1, 1, 2, 2], &device) * 0.123456789;
        let values = Tensor::<TestBackend, 4>::ones([1, 1, 1, 1], &device) * 0.987654321;
        let ranges = [0..1, 0..1, 0..1, 0..1];
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        
        // Check that precision is maintained
        let slice = result.clone().slice(ranges);
        let slice_data = slice.to_data();
        let slice_values = slice_data.as_slice::<f32>().unwrap();
        assert!((slice_values[0] - 0.987654321).abs() < 1e-7);
    }
}