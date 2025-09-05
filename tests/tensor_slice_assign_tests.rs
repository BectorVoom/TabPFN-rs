//! Comprehensive tests for tensor slice assignment functionality
//! 
//! This test suite validates the tensor slice assignment implementation across
//! different backends and ensures proper error handling, shape validation,
//! and numerical precision.

use burn::prelude::*;
use burn_ndarray::NdArray;

// Import the tensor slice assignment module
use tab_pfn_rs::tensor_slice_assign::*;

// Test backends
type CpuBackend = NdArray<f32>;

#[cfg(feature = "wgpu")]
use burn_wgpu::Wgpu;
#[cfg(feature = "wgpu")]
type GpuBackend = Wgpu<f32, i32>;

/// Helper function to create test tensors with specific values
fn create_tensor_with_value<B: Backend>(
    shape: [usize; 4], 
    value: f32, 
    device: &B::Device
) -> Tensor<B, 4> {
    if value == 0.0 {
        Tensor::zeros(shape, device)
    } else if value == 1.0 {
        Tensor::ones(shape, device)
    } else {
        Tensor::ones(shape, device) * value
    }
}

/// Test helper to compare tensor values with tolerance
fn tensors_equal<B: Backend>(
    a: &Tensor<B, 4>, 
    b: &Tensor<B, 4>, 
    tolerance: f32
) -> bool {
    if a.dims() != b.dims() {
        return false;
    }
    
    let a_data = a.to_data();
    let b_data = b.to_data();
    let a_values = a_data.as_slice::<f32>().expect("Failed to convert tensor a to f32");
    let b_values = b_data.as_slice::<f32>().expect("Failed to convert tensor b to f32");
    
    a_values.iter().zip(b_values.iter())
        .all(|(&a_val, &b_val)| (a_val - b_val).abs() <= tolerance)
}

/// Comprehensive test suite for CPU backend
mod cpu_tests {
    use super::*;
    
    #[test]
    fn test_normal_case_basic_4d_update() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([2, 3, 8, 8], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([2, 3, 4, 4], 1.0, &device);
        let ranges = [0..2, 0..3, 2..6, 2..6];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify the slice region has been updated
        assert!(validate_slice_values(&result, ranges.clone(), 1.0, 1e-6));
        
        // Verify unchanged regions remain 0.0
        assert!(validate_unchanged_regions(&tensor, &result, ranges, 1e-6));
    }
    
    #[test]
    fn test_normal_case_partial_channel_update() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([1, 4, 6, 6], 1.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([1, 2, 6, 6], 2.0, &device);
        let ranges = [0..1, 1..3, 0..6, 0..6];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify the updated channels
        assert!(validate_slice_values(&result, ranges.clone(), 2.0, 1e-6));
        
        // Verify other channels remain unchanged
        let ch0_ranges = [0..1, 0..1, 0..6, 0..6];
        let ch3_ranges = [0..1, 3..4, 0..6, 0..6];
        assert!(validate_slice_values(&result, ch0_ranges, 1.0, 1e-6));
        assert!(validate_slice_values(&result, ch3_ranges, 1.0, 1e-6));
    }
    
    #[test]
    fn test_normal_case_corner_slice() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([2, 2, 4, 4], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([1, 1, 2, 2], 3.0, &device);
        let ranges = [0..1, 0..1, 0..2, 0..2];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify corner region
        assert!(validate_slice_values(&result, ranges.clone(), 3.0, 1e-6));
        
        // Verify unchanged regions
        assert!(validate_unchanged_regions(&tensor, &result, ranges, 1e-6));
    }
    
    #[test]
    fn test_edge_case_single_element() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([1, 1, 1, 1], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([1, 1, 1, 1], 5.0, &device);
        let ranges = [0..1, 0..1, 0..1, 0..1];
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        
        assert!(validate_slice_values(&result, ranges, 5.0, 1e-6));
    }
    
    #[test]
    fn test_edge_case_full_tensor() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([2, 2, 3, 3], 1.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([2, 2, 3, 3], 7.0, &device);
        let ranges = [0..2, 0..2, 0..3, 0..3];
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        
        assert!(validate_slice_values(&result, ranges, 7.0, 1e-6));
    }
    
    #[test]
    fn test_edge_case_boundary_slices() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([2, 2, 4, 4], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([2, 2, 4, 2], 4.0, &device);
        let ranges = [0..2, 0..2, 0..4, 2..4]; // Right edge
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        assert!(validate_slice_values(&result, ranges.clone(), 4.0, 1e-6));
        assert!(validate_unchanged_regions(&tensor, &result, ranges, 1e-6));
    }
    
    #[test]
    fn test_error_case_shape_mismatch() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([2, 3, 8, 8], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([2, 3, 5, 5], 1.0, &device);
        let ranges = [0..2, 0..3, 2..6, 2..6]; // Expects [2,3,4,4]
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Shape mismatch"));
        assert!(error_msg.contains("[2, 3, 4, 4]"));
        assert!(error_msg.contains("[2, 3, 5, 5]"));
    }
    
    #[test]
    fn test_error_case_out_of_bounds() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([2, 3, 8, 8], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([3, 5, 9, 9], 1.0, &device);
        let ranges = [0..3, 0..5, 0..9, 0..9]; // Exceeds tensor dimensions
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Range out of bounds"));
        assert!(error_msg.contains("dimension 0"));
        assert!(error_msg.contains("exceeds size 2"));
    }
    
    #[test]
    fn test_error_case_invalid_range() {
        let device = Default::default();
        let tensor = create_tensor_with_value::<CpuBackend>([2, 3, 4, 4], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([1, 3, 4, 4], 1.0, &device);
        let ranges = [2..1, 0..3, 0..4, 0..4]; // Invalid: start > end
        
        let result = update_tensor_slice_4d(tensor, values, ranges);
        
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Invalid range"));
        assert!(error_msg.contains("start 2 >= end 1"));
        assert!(error_msg.contains("dimension 0"));
    }
    
    #[test]
    fn test_precision_preservation() {
        let device = Default::default();
        
        // Test with specific float values to ensure precision
        let tensor = create_tensor_with_value::<CpuBackend>([1, 1, 2, 2], 0.123456789, &device);
        let values = create_tensor_with_value::<CpuBackend>([1, 1, 1, 1], 0.987654321, &device);
        let ranges = [0..1, 0..1, 0..1, 0..1];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Check that precision is maintained
        let slice = result.clone().slice(ranges.clone());
        let slice_data = slice.to_data();
        let slice_values = slice_data.as_slice::<f32>().unwrap();
        assert!((slice_values[0] - 0.987654321).abs() < 1e-7);
        
        // Check unchanged region
        let unchanged_slice = result.slice([0..1, 0..1, 1..2, 1..2]);
        let unchanged_data = unchanged_slice.to_data();
        let unchanged_values = unchanged_data.as_slice::<f32>().unwrap();
        assert!((unchanged_values[0] - 0.123456789).abs() < 1e-7);
    }
}

/// Backend parity tests (CPU vs GPU when available)
#[cfg(feature = "wgpu")]
mod backend_parity_tests {
    use super::*;
    
    #[test]
    fn test_cpu_gpu_parity() {
        // CPU setup
        let cpu_device = Default::default();
        let cpu_tensor = create_tensor_with_value::<CpuBackend>([2, 3, 4, 4], 0.0, &cpu_device);
        let cpu_values = create_tensor_with_value::<CpuBackend>([2, 3, 2, 2], 1.0, &cpu_device);
        let ranges = [0..2, 0..3, 1..3, 1..3];
        
        // GPU setup
        let gpu_device = Default::default();
        let gpu_tensor = create_tensor_with_value::<GpuBackend>([2, 3, 4, 4], 0.0, &gpu_device);
        let gpu_values = create_tensor_with_value::<GpuBackend>([2, 3, 2, 2], 1.0, &gpu_device);
        
        // Perform operations
        let cpu_result = update_tensor_slice_4d(cpu_tensor, cpu_values, ranges.clone()).unwrap();
        let gpu_result = update_tensor_slice_4d(gpu_tensor, gpu_values, ranges.clone()).unwrap();
        
        // Convert GPU result to CPU for comparison
        let gpu_result_cpu = gpu_result.to_device(&cpu_device);
        
        // Compare results
        assert!(tensors_equal(&cpu_result, &gpu_result_cpu, 1e-6));
    }
    
    #[test]
    fn test_error_consistency_across_backends() {
        // Test that the same error conditions produce consistent error messages
        // across different backends
        
        let ranges = [0..3, 0..3, 0..8, 0..8]; // Out of bounds for [2,3,8,8] tensor
        
        // CPU error
        let cpu_device = Default::default();
        let cpu_tensor = create_tensor_with_value::<CpuBackend>([2, 3, 8, 8], 0.0, &cpu_device);
        let cpu_values = create_tensor_with_value::<CpuBackend>([3, 3, 8, 8], 1.0, &cpu_device);
        let cpu_result = update_tensor_slice_4d(cpu_tensor, cpu_values, ranges.clone());
        
        // GPU error
        let gpu_device = Default::default();
        let gpu_tensor = create_tensor_with_value::<GpuBackend>([2, 3, 8, 8], 0.0, &gpu_device);
        let gpu_values = create_tensor_with_value::<GpuBackend>([3, 3, 8, 8], 1.0, &gpu_device);
        let gpu_result = update_tensor_slice_4d(gpu_tensor, gpu_values, ranges);
        
        // Both should error with similar messages
        assert!(cpu_result.is_err());
        assert!(gpu_result.is_err());
        
        let cpu_error = cpu_result.unwrap_err();
        let gpu_error = gpu_result.unwrap_err();
        
        // Error messages should contain the same key information
        assert!(cpu_error.contains("Range out of bounds"));
        assert!(gpu_error.contains("Range out of bounds"));
        assert!(cpu_error.contains("dimension 0"));
        assert!(gpu_error.contains("dimension 0"));
    }
}

/// Performance and stress tests
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_large_tensor_slice_assignment() {
        let device = Default::default();
        
        // Test with a larger tensor to ensure scalability
        let tensor = create_tensor_with_value::<CpuBackend>([4, 8, 32, 32], 0.0, &device);
        let values = create_tensor_with_value::<CpuBackend>([2, 4, 16, 16], 1.0, &device);
        let ranges = [1..3, 2..6, 8..24, 8..24];
        
        let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone()).unwrap();
        
        // Verify correctness
        assert!(validate_slice_values(&result, ranges.clone(), 1.0, 1e-6));
        assert!(validate_unchanged_regions(&tensor, &result, ranges, 1e-6));
    }
    
    #[test]
    fn test_multiple_sequential_updates() {
        let device = Default::default();
        let mut tensor = create_tensor_with_value::<CpuBackend>([2, 2, 4, 4], 0.0, &device);
        
        // Apply multiple updates sequentially
        let updates = [
            ([0..1, 0..1, 0..2, 0..2], 1.0),
            ([0..1, 1..2, 0..2, 0..2], 2.0),
            ([1..2, 0..1, 0..2, 0..2], 3.0),
            ([1..2, 1..2, 0..2, 0..2], 4.0),
        ];
        
        for (ranges, value) in updates.iter() {
            let values = create_tensor_with_value::<CpuBackend>([1, 1, 2, 2], *value, &device);
            tensor = update_tensor_slice_4d(tensor, values, ranges.clone()).unwrap();
        }
        
        // Verify each quadrant has the correct value
        for (ranges, expected_value) in updates.iter() {
            assert!(validate_slice_values(&tensor, ranges.clone(), *expected_value, 1e-6));
        }
    }
}