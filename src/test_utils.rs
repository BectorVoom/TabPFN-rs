// Test utilities for backend-aware tensor construction
// 
// This module provides helper functions for creating tensors that work
// with Burn 0.18's Into<TensorData> trait bounds. All functions use
// Vec<T> + .as_slice() pattern to satisfy the API requirements.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Create f32 tensor from data slice with specified shape
/// 
/// Uses Vec-based approach to satisfy Burn 0.18 Into<TensorData> trait bounds.
/// Supports tensors of any dimensionality through reshape.
pub fn tensor_from_f32_vec<B: Backend, const D: usize>(
    data: &[f32], 
    shape: &[usize], 
    device: &B::Device
) -> Tensor<B, D> {
    // Verify data length matches shape
    let expected_size: usize = shape.iter().product();
    assert_eq!(data.len(), expected_size, 
        "Data length {} doesn't match shape {:?} (expected {})", 
        data.len(), shape, expected_size);
    
    // Use Vec + .as_slice() to satisfy Into<TensorData> for Burn 0.18
    let data_vec: Vec<f32> = data.to_vec();
    let flat_tensor = Tensor::<B, 1>::from_floats(data_vec.as_slice(), device);
    
    // Reshape using Burn's Shape type for arbitrary dimensions
    let shape_obj = burn::tensor::Shape::from(shape);
    flat_tensor.reshape(shape_obj)
}

/// Create i64 tensor from data slice with specified shape
/// 
/// Uses Vec-based approach to satisfy Burn 0.18 Into<TensorData> trait bounds.
pub fn tensor_from_i64_vec<B: Backend, const D: usize>(
    data: &[i64], 
    shape: &[usize], 
    device: &B::Device
) -> Tensor<B, D, burn::tensor::Int> {
    let expected_size: usize = shape.iter().product();
    assert_eq!(data.len(), expected_size, 
        "Data length {} doesn't match shape {:?} (expected {})", 
        data.len(), shape, expected_size);
    
    let data_vec: Vec<i64> = data.to_vec();
    let flat_tensor = Tensor::<B, 1, burn::tensor::Int>::from_ints(data_vec.as_slice(), device);
    let shape_obj = burn::tensor::Shape::from(shape);
    flat_tensor.reshape(shape_obj)
}

/// Create i32 tensor from data slice with specified shape  
/// 
/// Convenience function for i32 data, converts to i64 internally.
pub fn tensor_from_i32_vec<B: Backend, const D: usize>(
    data: &[i32], 
    shape: &[usize], 
    device: &B::Device
) -> Tensor<B, D, burn::tensor::Int> {
    let data_i64: Vec<i64> = data.iter().map(|&x| x as i64).collect();
    tensor_from_i64_vec(data_i64.as_slice(), shape, device)
}

/// Create bool tensor from data slice with specified shape
/// 
/// Uses Vec-based approach to satisfy Burn 0.18 Into<TensorData> trait bounds.
pub fn tensor_from_bool_vec<B: Backend, const D: usize>(
    data: &[bool], 
    shape: &[usize], 
    device: &B::Device
) -> Tensor<B, D, burn::tensor::Bool> {
    let expected_size: usize = shape.iter().product();
    assert_eq!(data.len(), expected_size, 
        "Data length {} doesn't match shape {:?} (expected {})", 
        data.len(), shape, expected_size);
    
    let data_vec: Vec<bool> = data.to_vec();
    let flat_tensor = Tensor::<B, 1, burn::tensor::Bool>::from_bool(data_vec.as_slice().into(), device);
    let shape_obj = burn::tensor::Shape::from(shape);
    flat_tensor.reshape(shape_obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_tensor_from_f32_vec_1d() {
        let device = Default::default();
        let data = [1.0f32, 2.0f32, 3.0f32];
        let tensor: Tensor<TestBackend, 1> = tensor_from_f32_vec(&data, &[3], &device);
        
        assert_eq!(tensor.dims(), [3]);
        let values: Vec<f32> = tensor.into_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_tensor_from_f32_vec_2d() {
        let device = Default::default();
        let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let tensor: Tensor<TestBackend, 2> = tensor_from_f32_vec(&data, &[2, 2], &device);
        
        assert_eq!(tensor.dims(), [2, 2]);
        let values: Vec<f32> = tensor.into_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_tensor_from_i64_vec() {
        let device = Default::default();
        let data = [1i64, 2i64, 3i64];
        let tensor: Tensor<TestBackend, 1, burn::tensor::Int> = tensor_from_i64_vec(&data, &[3], &device);
        
        assert_eq!(tensor.dims(), [3]);
        let values: Vec<i64> = tensor.into_data().to_vec().unwrap();
        assert_eq!(values, vec![1, 2, 3]);
    }
    
    #[test] 
    fn test_tensor_from_bool_vec() {
        let device = Default::default();
        let data = [true, false, true];
        let tensor: Tensor<TestBackend, 1, burn::tensor::Bool> = tensor_from_bool_vec(&data, &[3], &device);
        
        assert_eq!(tensor.dims(), [3]);
        let values: Vec<bool> = tensor.into_data().to_vec().unwrap();
        assert_eq!(values, vec![true, false, true]);
    }
    
    #[test]
    #[should_panic(expected = "Data length 2 doesn't match shape [3] (expected 3)")]
    fn test_mismatched_size_panics() {
        let device = Default::default();
        let data = [1.0f32, 2.0f32]; // 2 elements
        let _tensor: Tensor<TestBackend, 1> = tensor_from_f32_vec(&data, &[3], &device); // shape expects 3
    }
}